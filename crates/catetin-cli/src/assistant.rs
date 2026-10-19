use anyhow::Result;

pub fn query_llm(
    endpoint: &str,
    model: &str,
    api_key: Option<&str>,
    system_prompt: &str,
    user_prompt: &str,
) -> Result<String> {
    let url = if endpoint.ends_with("/chat/completions") {
        endpoint.to_string()
    } else if endpoint.ends_with('/') {
        format!("{}chat/completions", endpoint)
    } else {
        format!("{}/chat/completions", endpoint)
    };

    let payload = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    });

    let payload_str = serde_json::to_string(&payload)?;

    let curl_bin = if cfg!(windows) { "curl.exe" } else { "curl" };
    let mut cmd = std::process::Command::new(curl_bin);
    cmd.args(["-s", "-X", "POST", &url, "-H", "Content-Type: application/json"]);
    if let Some(key) = api_key {
        if !key.is_empty() {
            cmd.args(["-H", &format!("Authorization: Bearer {}", key)]);
        }
    }
    cmd.args(["-d", &payload_str]);

    let output = cmd
        .output()
        .map_err(|e| anyhow::anyhow!("Failed to run curl: {}", e))?;

    if !output.status.success() {
        let err_text = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!("LLM request failed: {}", err_text));
    }

    let response_str = String::from_utf8_lossy(&output.stdout);
    let val: serde_json::Value = serde_json::from_str(&response_str)
        .map_err(|_| anyhow::anyhow!("Invalid JSON response from LLM: {}", response_str))?;

    if let Some(content) = val["choices"][0]["message"]["content"].as_str() {
        Ok(content.trim().to_string())
    } else if let Some(err_msg) = val["error"]["message"].as_str() {
        Err(anyhow::anyhow!("LLM error: {}", err_msg))
    } else {
        Err(anyhow::anyhow!("Unknown LLM response shape"))
    }
}

pub fn generate_summary(
    endpoint: &str,
    model: &str,
    api_key: Option<&str>,
    transcript_text: &str,
) -> Result<(Vec<String>, Vec<String>)> {
    let sys = "You are an executive meeting copilot. Analyze the meeting transcript.\nReturn ONLY two sections formatted exactly like this:\nKEY TOPICS:\n- Topic 1\n- Topic 2\nACTION ITEMS:\n[ ] Person: Action 1\n[ ] Person: Action 2";
    let user = format!("Transcript:\n{}", transcript_text);

    let reply = query_llm(endpoint, model, api_key, sys, &user)?;

    let mut topics = Vec::new();
    let mut actions = Vec::new();
    let mut in_topics = false;
    let mut in_actions = false;

    for line in reply.lines() {
        let trimmed = line.trim();
        if trimmed.to_uppercase().contains("KEY TOPICS") {
            in_topics = true;
            in_actions = false;
        } else if trimmed.to_uppercase().contains("ACTION ITEMS") {
            in_actions = true;
            in_topics = false;
        } else if in_topics && !trimmed.is_empty() {
            let clean = trimmed.trim_start_matches(|c| c == '-' || c == '*' || c == '•').trim();
            if !clean.is_empty() {
                topics.push(clean.to_string());
            }
        } else if in_actions && !trimmed.is_empty() {
            let clean = trimmed.trim_start_matches(|c| c == '-' || c == '*' || c == '•').trim();
            if !clean.is_empty() {
                actions.push(clean.to_string());
            }
        }
    }

    Ok((topics, actions))
}

pub fn ask_assistant(
    endpoint: &str,
    model: &str,
    api_key: Option<&str>,
    transcript_text: &str,
    question: &str,
) -> Result<String> {
    let sys = "You are an AI meeting assistant. Answer the user's question directly based ONLY on the meeting transcript. Be concise, accurate, and direct.";
    let user = format!("Meeting Transcript:\n{}\n\nQuestion: {}", transcript_text, question);

    query_llm(endpoint, model, api_key, sys, &user)
}
