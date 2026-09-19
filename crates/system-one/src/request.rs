//! Parse and validate requests before they reach inference.

use crate::ValidationError;
use serde_json::{Map, Value, json};

#[derive(Clone, Debug)]
pub struct Request {
    pub(crate) model: String,
    pub(crate) state: Value,
    pub(crate) questions: Vec<(String, Question)>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum QuestionKind {
    Noul,
    Choice,
    Score,
}

#[derive(Clone, Debug)]
pub(crate) struct Question {
    pub kind: QuestionKind,
    pub instructions: Option<Value>,
    pub labels: Vec<String>,
    pub descriptions: Vec<Value>,
}

fn required<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    parent: &[&str],
) -> Result<&'a Value, ValidationError> {
    let mut path = parent.to_vec();
    path.push(key);
    object
        .get(key)
        .ok_or_else(|| ValidationError::new(&path, "Field required", "missing"))
}

fn content(value: &Value) -> bool {
    value.is_string() || value.is_object() || value.is_array()
}

impl Request {
    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn parse(value: Value) -> Result<Self, ValidationError> {
        let object = value.as_object().ok_or_else(|| {
            ValidationError::new(&["body"], "Expected a JSON object", "model_type")
        })?;
        let model = required(object, "model", &["body"])?
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                ValidationError::new(
                    &["body", "model"],
                    "Expected a nonempty model name",
                    "string_type",
                )
            })?
            .to_owned();
        let state = required(object, "state", &["body"])?;
        if !content(state) {
            return Err(ValidationError::new(
                &["body", "state"],
                "Expected a string, object, or array",
                "value_error",
            ));
        }
        let questions = required(object, "questions", &["body"])?
            .as_object()
            .filter(|q| !q.is_empty())
            .ok_or_else(|| {
                ValidationError::new(
                    &["body", "questions"],
                    "At least one question is required",
                    "value_error",
                )
            })?;
        for (key, option) in object {
            let supported = match key.as_str() {
                "state" | "model" | "questions" => true,
                "steps" | "samples" => option.is_null() || option.as_u64() == Some(1),
                "think" => option.is_null() || option.as_u64() == Some(0),
                "sequential" => option.is_null() || option.as_bool() == Some(false),
                "images" => option.is_null() || option.as_array().is_some_and(Vec::is_empty),
                _ => {
                    return Err(ValidationError::new(
                        &["body", key],
                        "Unknown field",
                        "extra_forbidden",
                    ));
                }
            };
            if !supported {
                return Err(ValidationError::new(
                    &["body", key],
                    "This service supports one text-only read: steps=1, samples=1, think=0, sequential=false, and no images",
                    "value_error",
                ));
            }
        }
        let questions = questions
            .iter()
            .map(|(id, value)| Ok((id.clone(), Question::parse(id, value)?)))
            .collect::<Result<_, ValidationError>>()?;
        Ok(Self {
            model,
            state: state.clone(),
            questions,
        })
    }
}

impl Question {
    fn parse(id: &str, value: &Value) -> Result<Self, ValidationError> {
        let path = ["body", "questions", id];
        let criteria_path = ["body", "questions", id, "criteria"];
        let object = value.as_object().ok_or_else(|| {
            ValidationError::new(&path, "Expected a question object", "model_type")
        })?;
        for key in object.keys() {
            if !["type", "instructions", "criteria"].contains(&key.as_str()) {
                return Err(ValidationError::new(
                    &["body", "questions", id, key],
                    "Unknown field",
                    "extra_forbidden",
                ));
            }
        }
        let instructions = object.get("instructions").filter(|v| !v.is_null()).cloned();
        if instructions.as_ref().is_some_and(|v| !content(v)) {
            return Err(ValidationError::new(
                &["body", "questions", id, "instructions"],
                "Expected a string, object, or array",
                "value_error",
            ));
        }
        let question_type = required(object, "type", &path)?.as_str();
        let (kind, labels, descriptions) = match question_type {
            Some("noul") => {
                let mut descriptions = vec![json!("yes"), json!("no")];
                if let Some(criteria) = object.get("criteria").filter(|v| !v.is_null()) {
                    let criteria = criteria.as_object().ok_or_else(|| {
                        ValidationError::new(
                            &criteria_path,
                            "Expected an object with true and false descriptions",
                            "dict_type",
                        )
                    })?;
                    if criteria.keys().any(|k| k != "true" && k != "false") {
                        return Err(ValidationError::new(
                            &criteria_path,
                            "Only true and false criteria are allowed",
                            "value_error",
                        ));
                    }
                    for (i, key) in ["true", "false"].iter().enumerate() {
                        if let Some(value) = criteria.get(*key) {
                            descriptions[i] = value.clone();
                        }
                    }
                }
                (
                    QuestionKind::Noul,
                    vec!["yes".into(), "no".into()],
                    descriptions,
                )
            }
            Some("choice") => {
                let criteria = required(object, "criteria", &path)?
                    .as_object()
                    .filter(|v| (1..=128).contains(&v.len()))
                    .ok_or_else(|| {
                        ValidationError::new(
                            &criteria_path,
                            "Expected 1 to 128 options",
                            "value_error",
                        )
                    })?;
                (
                    QuestionKind::Choice,
                    criteria.keys().cloned().collect(),
                    criteria.values().cloned().collect(),
                )
            }
            Some("score") => {
                let criteria = required(object, "criteria", &path)?
                    .as_array()
                    .filter(|v| (2..=10).contains(&v.len()) && v.iter().all(Value::is_string))
                    .ok_or_else(|| {
                        ValidationError::new(
                            &criteria_path,
                            "Expected 2 to 10 string levels, ordered lowest to highest",
                            "value_error",
                        )
                    })?;
                (
                    QuestionKind::Score,
                    (0..criteria.len()).map(|i| i.to_string()).collect(),
                    criteria.clone(),
                )
            }
            _ => {
                return Err(ValidationError::new(
                    &["body", "questions", id, "type"],
                    "Expected noul, choice, or score",
                    "literal_error",
                ));
            }
        };
        Ok(Self {
            kind,
            instructions,
            labels,
            descriptions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Value {
        json!({"model":"jev-latest", "state":"test", "questions":{"q":{"type":"noul"}}})
    }

    #[test]
    fn required_fields_and_invalid_types_have_precise_locations() {
        let error = Request::parse(json!({"model":"x"})).unwrap_err();
        assert_eq!(error.loc, vec![json!("body"), json!("state")]);
        assert_eq!(error.kind, "missing");
        let mut value = base();
        value["questions"]["q"]["type"] = json!("text");
        assert_eq!(
            Request::parse(value).unwrap_err().loc,
            vec![json!("body"), json!("questions"), json!("q"), json!("type")]
        );
        for state in [Value::Null, json!(42), json!(true)] {
            let mut value = base();
            value["state"] = state;
            assert!(Request::parse(value).is_err());
        }
    }

    #[test]
    fn unsupported_extensions_are_never_silently_ignored() {
        for (key, unsupported, supported) in [
            ("steps", json!(2), json!(1)),
            ("samples", json!(2), json!(1)),
            ("think", json!(8), json!(0)),
            ("sequential", json!(true), json!(false)),
            ("images", json!(["image"]), json!([])),
        ] {
            let mut value = base();
            value[key] = unsupported;
            assert!(Request::parse(value.clone()).is_err());
            value[key] = supported;
            assert!(Request::parse(value).is_ok());
        }
    }

    #[test]
    fn choice_and_score_enforce_documented_cardinality() {
        for count in [0, 1, 128, 129] {
            let mut value = base();
            let criteria: Map<String, Value> =
                (0..count).map(|i| (i.to_string(), Value::Null)).collect();
            value["questions"]["q"] = json!({"type":"choice", "criteria":criteria});
            assert_eq!(Request::parse(value).is_ok(), (1..=128).contains(&count));
        }
        for count in [1, 2, 10, 11] {
            let mut value = base();
            value["questions"]["q"] = json!({"type":"score", "criteria":vec!["level";count]});
            assert_eq!(Request::parse(value).is_ok(), (2..=10).contains(&count));
        }
    }
}
