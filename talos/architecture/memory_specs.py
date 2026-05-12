from __future__ import annotations

RF_SIZE_OPTIONS = [1024, 2048, 4096, 8192, 16384, 32768]
GB_SIZE_OPTIONS = [8192, 16384, 32768, 65536, 131072]

RF_BANDWIDTH_MIN_BITS = 8
GB_BANDWIDTH_MIN_BITS = 64
DRAM_BANDWIDTH_RANGE_BITS = (64, 2048)

_RF_BANDWIDTH_MAX_BITS_BY_SIZE = {
    1024: 32,
    2048: 64,
    4096: 64,
    8192: 128,
    16384: 128,
    32768: 256,
}

_GB_BANDWIDTH_MAX_BITS_BY_SIZE = {
    8192: 256,
    16384: 512,
    32768: 512,
    65536: 1024,
    131072: 1024,
}

_MIN_RF_BITS_FOR_CACTI_BANDWIDTH = {
    16: 512,
    32: 1024,
    64: 2048,
    128: 4096,
    256: 8192,
}


def bits_to_bytes(size_bits: int) -> int:
    return size_bits // 8


def bytes_to_bits(size_bytes: int) -> int:
    return size_bytes * 8


def derive_rf_bandwidth_max_bits(rf_size_bits: int) -> int:
    try:
        return _RF_BANDWIDTH_MAX_BITS_BY_SIZE[rf_size_bits]
    except KeyError as exc:
        valid = ", ".join(str(size) for size in RF_SIZE_OPTIONS)
        raise ValueError(
            f"Unknown RF size {rf_size_bits} bits. Expected one of: {valid}."
        ) from exc


def derive_gb_bandwidth_max_bits(gb_size_bits: int) -> int:
    try:
        return _GB_BANDWIDTH_MAX_BITS_BY_SIZE[gb_size_bits]
    except KeyError as exc:
        valid = ", ".join(str(size) for size in GB_SIZE_OPTIONS)
        raise ValueError(
            f"Unknown GB size {gb_size_bits} bits. Expected one of: {valid}."
        ) from exc


def validate_rf_cacti_compatibility(
    rf_size_bits: int,
    bandwidth_max_bits: int,
) -> None:
    if bandwidth_max_bits <= 16:
        minimum_rf_bits = _MIN_RF_BITS_FOR_CACTI_BANDWIDTH[16]
    else:
        try:
            minimum_rf_bits = _MIN_RF_BITS_FOR_CACTI_BANDWIDTH[bandwidth_max_bits]
        except KeyError as exc:
            valid = ", ".join(str(width) for width in sorted(_MIN_RF_BITS_FOR_CACTI_BANDWIDTH))
            raise ValueError(
                f"Unknown RF bandwidth_max {bandwidth_max_bits} bits. Expected one of: {valid}, or <= 16."
            ) from exc

    if rf_size_bits < minimum_rf_bits:
        raise ValueError(
            "RF size is too small for CACTI with the configured bandwidth_max."
        )
