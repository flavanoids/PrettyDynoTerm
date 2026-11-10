/// Normalized datacenter hardware VRAM capacity tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum DatacenterTier {
    /// AMD Instinct MI50 (16 GB HBM2)
    Mi50_16Gb,
    /// AMD Instinct MI100 (32 GB HBM2)
    Mi100_32Gb,
    /// NVIDIA Hopper H100 SXM (80 GB HBM3)
    H100_80Gb,
    /// NVIDIA Hopper H100 NVL (94 GB HBM3)
    H100Nvl_94Gb,
    /// NVIDIA Hopper H200 NVL (141 GB HBM3e)
    H200_141Gb,
    /// Other hardware capacity (e.g. consumer dev box or future hardware)
    Custom(u64),
}

impl DatacenterTier {
    pub fn from_capacity_mb(mb: u64) -> Self {
        match mb {
            14_000..=18_000 => Self::Mi50_16Gb,
            30_000..=36_000 => Self::Mi100_32Gb,
            75_000..=85_000 => Self::H100_80Gb,
            90_000..=98_000 => Self::H100Nvl_94Gb,
            135_000..=150_000 => Self::H200_141Gb,
            other => Self::Custom(other),
        }
    }

    pub fn nominal_capacity_mb(&self) -> u64 {
        match self {
            Self::Mi50_16Gb => 16_384,
            Self::Mi100_32Gb => 32_768,
            Self::H100_80Gb => 81_920,
            Self::H100Nvl_94Gb => 96_256,
            Self::H200_141Gb => 144_384,
            Self::Custom(mb) => *mb,
        }
    }
}

/// VRAM allocation plan for datacenter preset execution.
/// Ensures buffer allocations scale up to 75% of physical VRAM capacity while strictly
/// capping at <= 90% of live free memory to prevent host kernel OOM or VRAM over-commitment.
#[derive(Debug, Clone, PartialEq)]
pub struct VramAllocationPlan {
    pub tier: DatacenterTier,
    pub physical_capacity_bytes: u64,
    pub free_bytes: u64,
    pub max_allowed_bytes: u64,
    pub requested_bytes: u64,
    pub allocated_bytes: u64,
    pub allocation_ratio: f64,
    pub allocation_ratio_free: f64,
    pub is_clamped: bool,
    pub target_resolution: (u32, u32),
}

impl VramAllocationPlan {
    /// Computes allocation plan from live free memory and total capacity.
    /// Strictly enforces:
    /// 1. Never allocate above 90% of live free memory.
    /// 2. Never allocate above 75% of total physical capacity.
    pub fn compute_from_live(total_mb: u64, free_mb: u64, requested_bytes: u64) -> Self {
        let physical_capacity_bytes = total_mb.saturating_mul(1024 * 1024);
        let free_bytes = free_mb.saturating_mul(1024 * 1024);

        let max_from_free = free_bytes.saturating_mul(90) / 100;
        let max_from_total = physical_capacity_bytes.saturating_mul(75) / 100;
        let max_allowed_bytes = max_from_free.min(max_from_total);

        let (allocated_bytes, is_clamped) = if requested_bytes <= max_allowed_bytes {
            (requested_bytes, false)
        } else {
            (max_allowed_bytes, true)
        };

        let allocation_ratio = if physical_capacity_bytes > 0 {
            allocated_bytes as f64 / physical_capacity_bytes as f64
        } else {
            0.0
        };

        let allocation_ratio_free = if free_bytes > 0 {
            allocated_bytes as f64 / free_bytes as f64
        } else {
            0.0
        };

        let tier = DatacenterTier::from_capacity_mb(total_mb);
        let target_resolution = (15360, 8640);

        Self {
            tier,
            physical_capacity_bytes,
            free_bytes,
            max_allowed_bytes,
            requested_bytes,
            allocated_bytes,
            allocation_ratio,
            allocation_ratio_free,
            is_clamped,
            target_resolution,
        }
    }

    /// Computes an allocation plan assuming total capacity is free.
    pub fn compute(physical_capacity_mb: u64, requested_bytes: u64) -> Self {
        Self::compute_from_live(physical_capacity_mb, physical_capacity_mb, requested_bytes)
    }

    pub fn max_allowed_75pct_bytes(&self) -> u64 {
        self.physical_capacity_bytes.saturating_mul(75) / 100
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vram_tier_16gb_mi50() {
        let plan = VramAllocationPlan::compute(16_384, 15_000_000_000);
        assert!(plan.is_clamped);
        assert_eq!(
            plan.max_allowed_75pct_bytes(),
            16_384 * 1024 * 1024 * 75 / 100
        );
        assert_eq!(plan.allocated_bytes, plan.max_allowed_75pct_bytes());
        assert!(plan.allocation_ratio <= 0.75);
    }

    #[test]
    fn test_vram_tier_32gb_mi100() {
        let plan = VramAllocationPlan::compute(32_768, 20_000_000_000);
        assert!(!plan.is_clamped);
        assert_eq!(plan.allocated_bytes, 20_000_000_000);
        assert!(plan.allocation_ratio <= 0.75);

        // Over-allocation clamp test
        let plan_over = VramAllocationPlan::compute(32_768, 30_000_000_000);
        assert!(plan_over.is_clamped);
        assert_eq!(
            plan_over.allocated_bytes,
            plan_over.max_allowed_75pct_bytes()
        );
        assert!(plan_over.allocation_ratio <= 0.75);
    }

    #[test]
    fn test_vram_tier_80gb_h100() {
        let plan = VramAllocationPlan::compute(81_920, 100_000_000_000);
        assert!(plan.is_clamped);
        assert_eq!(
            plan.max_allowed_75pct_bytes(),
            81_920 * 1024 * 1024 * 75 / 100
        );
        assert_eq!(plan.allocated_bytes, 64_424_509_440); // 60 GB
        assert!(plan.allocation_ratio <= 0.75);
    }

    #[test]
    fn test_vram_tier_94gb_h100nvl() {
        let plan = VramAllocationPlan::compute(96_256, 100_000_000_000);
        assert!(plan.is_clamped);
        assert_eq!(plan.tier, DatacenterTier::H100Nvl_94Gb);
        assert_eq!(plan.allocated_bytes, 96_256 * 1024 * 1024 * 75 / 100);
        assert!(plan.allocation_ratio <= 0.75);
    }

    #[test]
    fn test_vram_tier_141gb_h200() {
        let plan = VramAllocationPlan::compute(144_384, 120_000_000_000);
        assert!(plan.is_clamped);
        assert_eq!(
            plan.max_allowed_75pct_bytes(),
            144_384 * 1024 * 1024 * 75 / 100
        );
        assert_eq!(plan.allocated_bytes, plan.max_allowed_75pct_bytes()); // 105.75 GB
        assert!(plan.allocation_ratio <= 0.75);
    }

    #[test]
    fn test_vram_live_free_clamp_90pct() {
        // GPU has 80 GB total, but only 20 GB free currently
        let plan = VramAllocationPlan::compute_from_live(81_920, 20_480, 25_000_000_000);
        assert!(plan.is_clamped);
        // Max from free = 20 GB * 0.90 = 18 GB = 19,327,352,832 bytes
        let max_90_free = 20_480 * 1024 * 1024 * 90 / 100;
        assert_eq!(plan.allocated_bytes, max_90_free);
        assert!(plan.allocation_ratio_free <= 0.90);
    }
}
