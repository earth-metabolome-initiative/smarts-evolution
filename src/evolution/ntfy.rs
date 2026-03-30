use std::path::Path;
use std::time::Duration;

use reqwest::blocking::Client;
use uuid::Uuid;

const DEFAULT_NTFY_BASE_URL: &str = "https://ntfy.sh";

#[derive(Clone, Debug)]
pub struct RunNotifier {
    topic_url: String,
    client: Client,
}

#[derive(Clone, Debug)]
pub struct ClassCompletionNotification<'a> {
    pub class_name: &'a str,
    pub best_smarts: &'a str,
    pub best_mcc: f64,
    pub elapsed: Duration,
    pub completed_generations: u64,
    pub node_id: usize,
    pub node_level: usize,
    pub completed_classes: u64,
    pub total_classes: u64,
    pub positive_count: usize,
    pub candidate_count: usize,
    pub total_test_molecules: usize,
    pub fold_count: usize,
    pub population_size: usize,
    pub worker_processes: usize,
    pub checkpoint_dir: &'a Path,
}

impl RunNotifier {
    pub fn new() -> Self {
        let topic_id = Uuid::new_v4();
        let base_url = std::env::var("SMARTS_EVOLUTION_NTFY_BASE_URL")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_NTFY_BASE_URL.to_string());
        let topic_url = format!("{}/{}", base_url.trim_end_matches('/'), topic_id);

        Self {
            topic_url,
            client: Client::new(),
        }
    }

    pub fn topic_url(&self) -> &str {
        &self.topic_url
    }

    pub fn notify_class_complete(
        &self,
        notification: &ClassCompletionNotification<'_>,
    ) -> Result<(), reqwest::Error> {
        let title = sanitize_header_value(&format!("Class complete: {}", notification.class_name));

        self.client
            .post(&self.topic_url)
            .header("Title", title)
            .header("Tags", "test_tube,white_check_mark")
            .header("Priority", "default")
            .header("Markdown", "yes")
            .body(notification_body(notification))
            .send()?
            .error_for_status()?;

        Ok(())
    }
}

fn notification_body(notification: &ClassCompletionNotification<'_>) -> String {
    format!(
        concat!(
            "# Class Tuning Complete\n\n",
            "**Class:** {}\n\n",
            "**Best SMARTS**\n",
            "```smarts\n{}\n```\n\n",
            "- MCC: `{:.4}`\n",
            "- Elapsed: `{}`\n",
            "- Generations: `{}`\n",
            "- Node: `{} / level {}`\n",
            "- Progress: `{}/{}` classes complete\n",
            "- Positive molecules: `{}`\n",
            "- Candidate molecules: `{}`\n",
            "- Fold test molecules: `{}` across `{}` folds\n",
            "- Population size: `{}`\n",
            "- Worker processes: `{}`\n",
            "- Checkpoints: `{}`\n"
        ),
        notification.class_name,
        notification.best_smarts,
        notification.best_mcc,
        format_duration(notification.elapsed),
        notification.completed_generations,
        notification.node_id,
        notification.node_level,
        notification.completed_classes,
        notification.total_classes,
        notification.positive_count,
        notification.candidate_count,
        notification.total_test_molecules,
        notification.fold_count,
        notification.population_size,
        notification.worker_processes,
        notification.checkpoint_dir.display(),
    )
}

fn sanitize_header_value(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            '\r' | '\n' => ' ',
            ch if ch.is_ascii() && !ch.is_ascii_control() => ch,
            _ => '?',
        })
        .collect()
}

fn format_duration(duration: Duration) -> String {
    let total_seconds = duration.as_secs();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;

    if hours > 0 {
        format!("{hours}h {minutes:02}m {seconds:02}s")
    } else if minutes > 0 {
        format!("{minutes}m {seconds:02}s")
    } else {
        format!("{seconds}.{:03}s", duration.subsec_millis())
    }
}

#[cfg(test)]
mod tests {
    use super::{ClassCompletionNotification, format_duration, notification_body};
    use std::path::Path;
    use std::time::Duration;

    #[test]
    fn duration_format_is_compact_and_readable() {
        assert_eq!(format_duration(Duration::from_millis(987)), "0.987s");
        assert_eq!(format_duration(Duration::from_secs(75)), "1m 15s");
        assert_eq!(format_duration(Duration::from_secs(3723)), "1h 02m 03s");
    }

    #[test]
    fn notification_body_includes_key_class_details() {
        let notification = ClassCompletionNotification {
            class_name: "Penicillins",
            best_smarts: "[#6]-[#7]",
            best_mcc: 0.8562,
            elapsed: Duration::from_secs(83),
            completed_generations: 17,
            node_id: 42,
            node_level: 2,
            completed_classes: 5,
            total_classes: 17,
            positive_count: 111,
            candidate_count: 572,
            total_test_molecules: 301,
            fold_count: 5,
            population_size: 256,
            worker_processes: 12,
            checkpoint_dir: Path::new("checkpoints/npc"),
        };

        let body = notification_body(&notification);

        assert!(body.contains("# Class Tuning Complete"));
        assert!(body.contains("**Class:** Penicillins"));
        assert!(body.contains("```smarts\n[#6]-[#7]\n```"));
        assert!(body.contains("- MCC: `0.8562`"));
        assert!(body.contains("- Progress: `5/17` classes complete"));
        assert!(body.contains("- Checkpoints: `checkpoints/npc`"));
    }
}
