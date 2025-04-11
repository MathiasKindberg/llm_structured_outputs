static CLIENT: std::sync::LazyLock<reqwest::Client> =
    std::sync::LazyLock::new(reqwest::Client::new);
static CONFIG: std::sync::LazyLock<Config> = std::sync::LazyLock::new(Config::new);

/// Create an OpenAI compatible schema from a Rust type. Utilizes a bastardized version of the
/// structured outputs type's name for the schema name sent to OpenAI.
pub fn get_schema<T: schemars::JsonSchema>() -> Schema {
    let mut schema = schemars::generate::SchemaSettings::default()
        .for_serialize()
        .with(|s| s.meta_schema = None)
        // The schema generator automatically adds "format" to the items specifying for example int64
        // or double. OpenAI does not support this.
        .with_transform(schemars::transform::RecursiveTransform(
            |schema: &mut schemars::Schema| {
                schema.remove("format");
            },
        ))
        .into_generator()
        .into_root_schema_for::<T>();

    // Remove title field from schema since OpenAI api does not support it.
    schema.as_object_mut().unwrap().remove("title");

    // We need a name for the schema. Get the type name and ensure it
    // is compatible with OpenAI as per the regex "^[a-zA-Z0-9_-]+$"
    let name = std::any::type_name::<T>()
        .replace("::", "_")
        .replace("<", "_")
        .replace(">", "_");

    Schema {
        name,
        schema: serde_json::to_value(schema).expect("Failed to convert schema to JSON value"),
        strict: true,
    }
}

/// Query OpenAI with a message and a schema. The schema is used to enforce structured output
/// from the OpenAI API and parse the response into said Rust type.
pub async fn query_openai<T>(messages: Vec<Message>) -> T
where
    T: for<'a> serde::Deserialize<'a> + schemars::JsonSchema,
{
    let query = OpenAIChatCompletionQuery {
        model: CONFIG.model.clone(),
        messages,
        response_format: ResponseFormat {
            response_type: "json_schema".to_string(),
            json_schema: get_schema::<T>(),
        },
    };

    let response = CLIENT
        .post("https://api.openai.com/v1/chat/completions")
        // OpenAI requires non-standard format for the Authorization header....
        // .bearer_auth(...) does not work...
        .header(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", CONFIG.api_key),
        )
        .json(&query)
        .send()
        .await
        .expect("Response from OpenAI");

    // If OpenAI returns an error print the raw output for debugging.
    if let Err(e) = response.error_for_status_ref() {
        panic!(
            "Error querying api: {e}\nRaw output:\n{}",
            response.text().await.unwrap()
        );
    }

    // The response is inside a string field, so we first need to parse the
    // entire response and then pick out the content field to parse separately
    // into our structured output type.
    let model_response: OpenAIChatCompletionResponse = response
        .json::<OpenAIChatCompletionResponse>()
        .await
        .expect("Response body");

    // Parse the first response into our structured output type.
    serde_json::from_str(
        &model_response
            .choices
            .get(0)
            .expect("Response from OpenAI")
            .message
            .content,
    )
    .expect("Parseable response")
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

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Message {
    role: Role,
    content: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum Role {
    Developer,
    User,
    Assistant,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAIChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, serde::Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Debug, serde::Deserialize)]
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

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
    #[serde(deny_unknown_fields)]
    struct SimpleResponseSchema {
        #[schemars(required)]
        #[schemars(description = "Summary of the text")]
        summary: String,

        #[schemars(required)]
        #[schemars(description = "Tone of the text")]
        tone: String,

        #[schemars(required)]
        #[schemars(description = "Number of words in the text")]
        word_count: i64,

        #[schemars(required)]
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
    #[serde(deny_unknown_fields)]
    struct NestedResponseSchema {
        #[schemars(required)]
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
            assert!(response.tone.len() > 0);
            assert!(response.word_count > 0);
            assert!(response.flair >= 0.0 && response.flair <= 1.0);
        }
    }
}
