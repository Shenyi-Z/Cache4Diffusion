import re

def convert_flops(flops_str):
    """
    Convert a FLOPS string (e.g., '12.34 GFLOPS', '1.2 TFLOPS') into the corresponding numerical value.
    """
    # Use regular expressions to match numbers and units
    match = re.match(r"([\d.]+)\s*([GT]?FLOPS)", flops_str.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Unable to parse FLOPS string: {flops_str}")
    
    # Extract the numeric value and unit
    value = float(match.group(1))
    unit = match.group(2).upper()
    
    # Convert based on the unit
    if unit == "GFLOPS":
        return value * 10**9
    elif unit == "TFLOPS":
        return value * 10**12
    else:
        raise ValueError(f"Unknown FLOPS unit: {unit}")
