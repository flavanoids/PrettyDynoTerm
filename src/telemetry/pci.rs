use serde::{Deserialize, Serialize};
use std::fmt;

/// Canonical PCI Bus Identifier, normalized to `[domain]:bus:device.function`
/// Example: `0000:03:00.0`
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub struct PciBusId(String);

impl PciBusId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(Self::normalize(&id.into()))
    }

    /// Normalizes a PCI bus ID string into `0000:bb:dd.f` format.
    /// Handles:
    /// - `00000000:03:00.0` (8-digit NVML domain) -> `0000:03:00.0`
    /// - `0000:03:00.0`     (standard 4-digit domain) -> `0000:03:00.0`
    /// - `03:00.0`          (missing domain) -> `0000:03:00.0`
    pub fn normalize(raw: &str) -> String {
        let trimmed = raw.trim().to_ascii_lowercase();
        if trimmed.is_empty() {
            return String::new();
        }

        let parts: Vec<&str> = trimmed.split(':').collect();
        match parts.len() {
            // "03:00.0" -> domain "0000", bus "03", dev.fn "00.0"
            2 => {
                let bus = parts[0];
                let dev_fn = parts[1];
                format!("0000:{:0>2}:{}", bus, dev_fn)
            }
            // "0000:03:00.0" or "00000000:03:00.0"
            3 => {
                let domain = parts[0];
                let bus = parts[1];
                let dev_fn = parts[2];

                // If domain has 8 chars, take last 4 if prefix is 0
                let normalized_domain = if domain.len() > 4 {
                    &domain[domain.len() - 4..]
                } else {
                    domain
                };

                format!("{:0>4}:{:0>2}:{}", normalized_domain, bus, dev_fn)
            }
            _ => trimmed,
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for PciBusId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Debug for PciBusId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PciBusId({})", self.0)
    }
}

impl From<&str> for PciBusId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for PciBusId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pci_bus_id_normalization() {
        assert_eq!(PciBusId::new("0000:03:00.0").as_str(), "0000:03:00.0");
        assert_eq!(PciBusId::new("00000000:03:00.0").as_str(), "0000:03:00.0");
        assert_eq!(PciBusId::new("03:00.0").as_str(), "0000:03:00.0");
        assert_eq!(PciBusId::new("3:00.0").as_str(), "0000:03:00.0");
        assert_eq!(PciBusId::new("0000:41:00.0").as_str(), "0000:41:00.0");
        assert_eq!(PciBusId::new("  0000:0A:00.0  ").as_str(), "0000:0a:00.0");
    }
}
