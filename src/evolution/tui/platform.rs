#[cfg(target_os = "linux")]
use std::fs;

pub(super) fn current_process_resident_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = fs::read_to_string("/proc/self/status").ok()?;
        status.lines().find_map(parse_vm_rss_line)
    }

    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(super) fn parse_vm_rss_line(line: &str) -> Option<u64> {
    let value = line.strip_prefix("VmRSS:")?.split_whitespace().next()?;
    value
        .parse::<u64>()
        .ok()
        .map(|kib| kib.saturating_mul(1024))
}
