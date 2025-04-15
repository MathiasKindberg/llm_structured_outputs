static CLIENT: std::sync::LazyLock<reqwest::Client> =
    std::sync::LazyLock::new(reqwest::Client::new);
static CONFIG: std::sync::LazyLock<Config> = std::sync::LazyLock::new(Config::new);

/// Create an OpenAI compatible schema from a Rust type. Utilizes a diagnostic version of the
/// desired response schema's type name for the schema name sent to OpenAI.
pub fn get_schema<T: schemars::JsonSchema>() -> Schema {
    let schema = schemars::generate::SchemaSettings::default()
        // The schema generator automatically adds "format" to the items specifying
        // for example int64 or double.
        // OpenAI does not support this.
        .with_transform(schemars::transform::RecursiveTransform(
            |schema: &mut schemars::Schema| {
                schema.remove("format");
            },
        ))
        .into_generator()
        .into_root_schema_for::<T>();
    let schema = serde_json::to_value(schema).expect("Failed to convert schema to JSON value");

    // We need a name for the schema. Get the type name and ensure it
    // is compatible with OpenAI as per the regex "^[a-zA-Z0-9_-]+$"
    let name = std::any::type_name::<T>()
        .to_string()
        .replace("::", "_")
        .replace("<", "_")
        .replace(">", "_");

    Schema {
        name,
        schema,
        strict: true,
    }
}

/// Query OpenAI with a message and a schema defined by the generic type T. The schema
/// is used to enforce structured output from the OpenAI API and parse the response into
/// said Rust type.
pub async fn query_openai<T>(messages: Vec<Message>) -> T
where
    T: for<'a> serde::Deserialize<'a> + schemars::JsonSchema,
{
    let schema = get_schema::<T>();
    let response = query_openai_inner(messages, schema)
        .await
        .expect("Response from OpenAI");

    // The response is inside a string field, so we first need to parse the
    // entire response and then pick out the content field to parse separately
    // into our structured output type.
    serde_json::from_str(
        &response
            .choices
            .get(0)
            .expect("Response from OpenAI")
            .message
            .content,
    )
    .expect("Correctly structured parseable response")
}

/// Query the OpenAI API with a message and a schema.
async fn query_openai_inner(
    messages: Vec<Message>,
    schema: Schema,
) -> anyhow::Result<OpenAIChatCompletionResponse> {
    let query = OpenAIChatCompletionQuery {
        model: CONFIG.model.clone(), // E.g. "o3-mini-2025-01-31"
        messages,
        response_format: ResponseFormat {
            // Always set to json_schema when using structured outputs
            response_type: "json_schema".to_string(),
            json_schema: schema,
        },
    };

    let response = CLIENT
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(CONFIG.api_key.clone())
        .json(&query)
        .send()
        .await?;

    if let Err(e) = response.error_for_status_ref() {
        anyhow::bail!(
            "Error querying api: {e}\nRaw output:\n{}",
            response.text().await.unwrap()
        );
    }

    Ok(response.json().await?)
}

#[derive(Debug, serde::Serialize)]
struct OpenAIChatCompletionQuery {
    model: String,
    messages: Vec<Message>,
    response_format: ResponseFormat,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    response_type: String,
    json_schema: Schema,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Schema {
    name: String,
    schema: serde_json::Value,
    strict: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Message {
    role: Role,
    content: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum Role {
    Developer,
    User,
    Assistant,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct OpenAIChatCompletionResponse {
    choices: Vec<Choice>,
    // There are a bunch of extra fields in the response
    // that we don't care about. See the OpenAI API docs.
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct ResponseMessage {
    content: String,
}

struct Config {
    api_key: String,
    model: String,
}

impl Config {
    fn new() -> Self {
        dotenvy::dotenv().expect("Failed to load .env file");
        Self {
            api_key: std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"),
            model: std::env::var("OPENAI_MODEL").expect("OPENAI_MODEL not set"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
    #[serde(deny_unknown_fields)]
    struct SimpleResponseSchema {
        #[schemars(description = "Summary of the text in two extremely short paragraphs")]
        summary: Vec<String>,

        #[schemars(description = "Tone of the text")]
        tone: String,

        #[schemars(description = "Number of words in the text")]
        word_count: i64,

        #[schemars(description = "Flair from 0 to 1")]
        flair: f64,
    }

    #[tokio::test]
    async fn test_simple_schema() {
        let response = query_openai::<SimpleResponseSchema>(vec![Message {
            role: Role::User,
            content: "Hello, world!".to_string(),
        }])
        .await;

        assert!(response.summary.len() > 0);
        assert!(response.tone.len() > 0);
        assert!(response.word_count > 0);
        assert!(response.flair >= 0.0 && response.flair <= 1.0);
    }

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
    enum Sentiment {
        Positive,
        Neutral,
        Negative,
    }

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
    #[serde(deny_unknown_fields)]
    struct EnumResponseSchema {
        #[schemars(description = "Summary of the text")]
        summary: String,

        #[schemars(description = "Sentiment of the text")]
        sentiment: Sentiment,

        #[schemars(description = "Number of words in the text")]
        word_count: i64,
    }

    #[tokio::test]
    async fn test_enum_schema() {
        let response = query_openai::<EnumResponseSchema>(vec![Message {
            role: Role::User,
            content: "I'm having a wonderful day today!".to_string(),
        }])
        .await;

        assert!(response.summary.len() > 0);
        assert!(response.word_count > 0);
        // No need to validate the enum sentiment since the query will fail if it is not
        // one of the values.
    }

    #[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
    #[serde(deny_unknown_fields)]
    struct NestedResponseSchema {
        #[schemars(description = "A list of responses to the message")]
        responses: Vec<SimpleResponseSchema>,
    }

    #[tokio::test]
    async fn test_nested_schema() {
        let responses = query_openai::<NestedResponseSchema>(vec![Message {
            role: Role::User,
            content: "Hello, world! Reply with at least 3 different responses".to_string(),
        }])
        .await;
        assert!(responses.responses.len() >= 3);

        for response in responses.responses {
            assert!(response.summary.len() > 0);
            // assert!(response.tone.len() > 0);
            // assert!(response.word_count > 0);
            // assert!(response.flair >= 0.0 && response.flair <= 1.0);
        }
    }

    /// Represents a complex schema that tests most features of the schema.
    #[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
    #[serde(deny_unknown_fields)]
    struct ComplexResponseSchema {
        #[schemars(description = "Summary of the text")]
        summary: String,

        #[schemars(description = "Number of words in the text")]
        word_count: i64,

        #[schemars(description = "Flair from 0 to 1")]
        flair: f64,

        #[schemars(description = "A list of responses to the message")]
        responses: Vec<SimpleResponseSchema>,

        #[schemars(description = "A list of sentiments to the message")]
        sentiments: Vec<Sentiment>,

        // No description is allowed when the type is an object
        object_in_object: SimpleResponseSchema,
    }

    #[tokio::test]
    async fn test_schemars_default_schema() {
        let schema = Schema {
            name: "ComplexResponseSchema".to_string(),
            schema: serde_json::to_value(schemars::schema_for!(ComplexResponseSchema))
                .expect("Failed to convert schema to JSON value"),
            strict: true,
        };

        let messages = vec![Message {
            role: Role::User,
            content: "Hello, world!".to_string(),
        }];

        // Now lets start getting it to work.
        let response = query_openai_inner(messages.clone(), schema).await;
        assert!(response.is_err());

        // Error querying api: HTTP status client error (400 Bad Request) for url (https://api.openai.com/v1/chat/completions)
        // Raw output:
        // {
        //   "error": {
        //     "message": "Invalid schema for response_format 'ComplexResponseSchema': In context=('properties', 'flair'), 'format' is not permitted.",
        //     "type": "invalid_request_error",
        //     "param": "response_format",
        //     "code": null
        //   }
        // }

        let schema = schemars::generate::SchemaSettings::default()
            .with_transform(schemars::transform::RecursiveTransform(
                |schema: &mut schemars::Schema| {
                    schema.remove("format");
                },
            ))
            .into_generator()
            .into_root_schema_for::<ComplexResponseSchema>();

        let schema = Schema {
            name: "ComplexResponseSchema".to_string(),
            schema: serde_json::to_value(schema).expect("Failed to convert schema to JSON value"),
            strict: true,
        };

        let response = query_openai_inner(messages, schema).await.unwrap();
        let response: ComplexResponseSchema = serde_json::from_str(
            &response
                .choices
                .get(0)
                .expect("Response from OpenAI")
                .message
                .content,
        )
        .unwrap();

        // Assert all fields in ComplexResponseSchema
        assert!(!response.summary.is_empty(), "Summary should not be empty");
        assert!(response.word_count > 0, "Word count should be positive");
        assert!(
            response.flair >= 0.0 && response.flair <= 1.0,
            "Flair should be between 0 and 1"
        );
        assert!(
            !response.responses.is_empty(),
            "Responses should not be empty"
        );
        assert!(
            !response.sentiments.is_empty(),
            "Sentiments should not be empty"
        );
        assert!(
            !response.object_in_object.summary.is_empty(),
            "Object in object summary should not be empty"
        );
    }
}
