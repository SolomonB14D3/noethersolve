#!/usr/bin/env python3
"""
Atmospheric Electricity Fact-Checker

Analyzes claims about extracting electrical energy from the atmosphere.

THE PHYSICS IS REAL:
- Earth has a net negative charge (~500,000 C)
- Ionosphere is positive (~300 kV potential difference)
- Fair-weather electric field: ~100-150 V/m at ground level
- Global atmospheric circuit: ~1-2 kW total power (from lightning)

THE LIMITATIONS ARE SEVERE:
- Energy density is extremely low
- Current is in the picoampere range per square meter
- Lightning is destructive and unpredictable
- Practical extraction is inefficient

This is NOT free energy - it comes from the global thunderstorm circuit.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math

# Physical constants for atmospheric electricity
EARTH_SURFACE_FIELD = 130  # V/m (fair-weather, ground level)
IONOSPHERE_POTENTIAL = 300_000  # V (300 kV above ground)
EARTH_CHARGE = 500_000  # Coulombs (approximate)
FAIR_WEATHER_CURRENT_DENSITY = 2e-12  # A/m² (2 pA/m²)
GLOBAL_CIRCUIT_CURRENT = 1800  # A (total fair-weather current)
GLOBAL_CIRCUIT_POWER = 400_000  # W (total, from ~2000 thunderstorms)
LIGHTNING_ENERGY = 1e9  # J per lightning bolt (approximate)
LIGHTNING_FREQUENCY = 100  # strikes per second globally

EARTH_SURFACE_AREA = 5.1e14  # m²


@dataclass
class AtmosphericEnergyDevice:
    """A proposed atmospheric energy harvesting device."""
    name: str
    description: str
    collector_area_m2: float
    collector_height_m: float
    claimed_power_watts: Optional[float]
    mechanism: str


def calculate_theoretical_power(device: AtmosphericEnergyDevice) -> Dict:
    """
    Calculate theoretical power available from atmospheric electricity.

    Power = Current × Voltage
    Current = current_density × area
    Voltage = field × height
    """
    # Fair-weather current through collector area
    current = FAIR_WEATHER_CURRENT_DENSITY * device.collector_area_m2

    # Voltage from ground to collector height
    # Field decreases with altitude, but approximate as constant for low heights
    voltage = EARTH_SURFACE_FIELD * device.collector_height_m

    # Theoretical power (optimistic - assumes perfect collection)
    theoretical_power = current * voltage

    # More realistic: field decreases, losses occur
    # At 100m, field is ~50 V/m; at 1km, ~30 V/m
    if device.collector_height_m > 100:
        avg_field = EARTH_SURFACE_FIELD * (1 - 0.3 * math.log10(device.collector_height_m / 10))
        realistic_voltage = avg_field * device.collector_height_m
        realistic_power = current * realistic_voltage * 0.1  # 10% efficiency
    else:
        realistic_power = theoretical_power * 0.1

    return {
        "collector_area_m2": device.collector_area_m2,
        "collector_height_m": device.collector_height_m,
        "fair_weather_current_A": current,
        "voltage_V": voltage,
        "theoretical_power_W": theoretical_power,
        "realistic_power_W": realistic_power,
    }


def analyze_atmospheric_claim(device: AtmosphericEnergyDevice) -> Dict:
    """
    Analyze an atmospheric energy claim against physics.
    """
    physics = calculate_theoretical_power(device)

    results = {
        "device": device.name,
        "physics": physics,
        "checks": [],
        "verdict": "PLAUSIBLE",
        "notes": []
    }

    # Check 1: Is claimed power physically possible?
    if device.claimed_power_watts is not None:
        theoretical = physics["theoretical_power_W"]
        ratio = device.claimed_power_watts / theoretical if theoretical > 0 else float('inf')

        if ratio > 100:
            results["checks"].append({
                "name": "Power claim vs theory",
                "passed": False,
                "message": f"Claimed {device.claimed_power_watts:.2e} W is {ratio:.0f}× theoretical maximum ({theoretical:.2e} W)"
            })
            results["verdict"] = "IMPLAUSIBLE"
        elif ratio > 10:
            results["checks"].append({
                "name": "Power claim vs theory",
                "passed": False,
                "message": f"Claimed power is {ratio:.0f}× theoretical - optimistic"
            })
            results["verdict"] = "QUESTIONABLE"
        else:
            results["checks"].append({
                "name": "Power claim vs theory",
                "passed": True,
                "message": f"Claimed power within theoretical bounds (ratio {ratio:.1f})"
            })

    # Check 2: Energy source identification
    mechanism = device.mechanism.lower()
    if "free energy" in mechanism or "zero point" in mechanism:
        results["checks"].append({
            "name": "Energy source",
            "passed": False,
            "message": "Atmospheric electricity is NOT free energy - comes from global thunderstorm circuit"
        })
        results["verdict"] = "MISLEADING"
    elif "lightning" in mechanism:
        results["checks"].append({
            "name": "Energy source",
            "passed": True,
            "message": "Lightning harvesting is theoretically possible but practically difficult"
        })
        results["notes"].append("Lightning is destructive and unpredictable")
    else:
        results["checks"].append({
            "name": "Energy source",
            "passed": True,
            "message": "Fair-weather atmospheric current is the energy source"
        })

    # Check 3: Practical considerations
    realistic = physics["realistic_power_W"]
    if realistic < 1e-6:  # Less than 1 microwatt
        results["checks"].append({
            "name": "Practicality",
            "passed": False,
            "message": f"Realistic power ~{realistic:.2e} W is below useful threshold"
        })
        if results["verdict"] == "PLAUSIBLE":
            results["verdict"] = "IMPRACTICAL"
    elif realistic < 1:  # Less than 1 watt
        results["checks"].append({
            "name": "Practicality",
            "passed": True,
            "message": f"Realistic power ~{realistic:.2e} W - may be useful for low-power sensors"
        })
    else:
        results["checks"].append({
            "name": "Practicality",
            "passed": True,
            "message": f"Realistic power ~{realistic:.2e} W - potentially useful"
        })

    # Add educational notes
    results["notes"].extend([
        f"Global atmospheric circuit provides ~{GLOBAL_CIRCUIT_POWER/1000:.0f} kW total",
        f"Fair-weather current density: {FAIR_WEATHER_CURRENT_DENSITY*1e12:.0f} pA/m²",
        f"Ground-level field: ~{EARTH_SURFACE_FIELD} V/m",
    ])

    return results


def print_analysis(results: Dict):
    """Pretty-print analysis results."""
    print("=" * 70)
    print(f"DEVICE: {results['device']}")
    print("=" * 70)

    physics = results["physics"]
    print(f"\nPHYSICS CALCULATION:")
    print(f"  Collector area: {physics['collector_area_m2']:.1f} m²")
    print(f"  Collector height: {physics['collector_height_m']:.1f} m")
    print(f"  Fair-weather current: {physics['fair_weather_current_A']:.2e} A")
    print(f"  Voltage: {physics['voltage_V']:.0f} V")
    print(f"  Theoretical power: {physics['theoretical_power_W']:.2e} W")
    print(f"  Realistic power: {physics['realistic_power_W']:.2e} W")

    print(f"\nCHECKS:")
    for check in results["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['name']}: {check['message']}")

    print(f"\nVERDICT: {results['verdict']}")

    if results["notes"]:
        print(f"\nNOTES:")
        for note in results["notes"]:
            print(f"  - {note}")

    print("=" * 70)


# ============================================================================
# Example Devices
# ============================================================================

ATMOSPHERIC_DEVICES = {
    "tesla_tower": AtmosphericEnergyDevice(
        name="Tesla's Wardenclyffe Tower (claimed)",
        description="Tall tower to collect atmospheric electricity",
        collector_area_m2=100,  # Approximate dome area
        collector_height_m=57,  # Actual tower height
        claimed_power_watts=10_000,  # Tesla's claims were vague
        mechanism="Collect atmospheric electricity and transmit wirelessly"
    ),

    "small_antenna": AtmosphericEnergyDevice(
        name="Small Atmospheric Antenna",
        description="Typical YouTube 'free energy' demo",
        collector_area_m2=0.01,  # 10 cm² wire
        collector_height_m=2,    # 2 meter antenna
        claimed_power_watts=10,  # Common claim
        mechanism="Free energy from atmospheric electricity"
    ),

    "large_balloon": AtmosphericEnergyDevice(
        name="High-Altitude Balloon Collector",
        description="Serious research proposal",
        collector_area_m2=1000,  # Large conducting surface
        collector_height_m=1000, # 1 km altitude
        claimed_power_watts=None,  # No exaggerated claim
        mechanism="Fair-weather current collection at altitude"
    ),

    "lightning_rod_harvester": AtmosphericEnergyDevice(
        name="Lightning Harvesting Tower",
        description="Attempt to capture lightning energy",
        collector_area_m2=10,
        collector_height_m=100,
        claimed_power_watts=1_000_000,  # MW from lightning
        mechanism="Lightning strikes captured and stored"
    ),

    "rf_harvester": AtmosphericEnergyDevice(
        name="RF Energy Harvester",
        description="Harvests radio waves from atmosphere",
        collector_area_m2=0.1,
        collector_height_m=1,
        claimed_power_watts=0.001,  # 1 mW
        mechanism="Rectenna converts RF to DC - legitimate but low power"
    ),

    "plauson_converter": AtmosphericEnergyDevice(
        name="Plauson Atmospheric Converter (1920s)",
        description="Hermann Plauson's patented device",
        collector_area_m2=500,  # Large collector
        collector_height_m=300, # Tall mast
        claimed_power_watts=100,  # Modest claim
        mechanism="Atmospheric electricity via elevated conductors"
    ),
}


def compare_to_alternatives():
    """Compare atmospheric electricity to other energy sources."""
    print("\n" + "=" * 70)
    print("COMPARISON: Atmospheric Electricity vs Other Sources")
    print("=" * 70)

    print("""
Energy Source          Power Density       Notes
---------------------------------------------------------------------------
Solar (ground)         ~1000 W/m²          Excellent, mature technology
Wind (good site)       ~500 W/m²           Good, requires large turbines
Atmospheric electric   ~2×10⁻¹⁰ W/m²       5 billion times weaker than solar!
RF harvesting          ~10⁻⁶ W/m²          Useful for low-power sensors
Lightning (averaged)   ~0.0008 W/m²        Destructive, unpredictable

CONCLUSION:
Atmospheric electricity is REAL but the power density is ~10 billion times
lower than solar. A 1 m² solar panel produces ~200W. To get the same from
atmospheric electricity, you'd need ~10¹⁰ m² of collector area.

This is why "atmospheric energy" devices are not practical for power
generation. However, they ARE useful for:
- Educational demonstrations
- Ultra-low-power sensors
- Franklin's lightning rod (safety, not energy)
""")


def main():
    """Analyze all atmospheric energy devices."""
    print("\nATMOSPHERIC ELECTRICITY FACT-CHECKER")
    print("Based on real atmospheric physics\n")

    for name, device in ATMOSPHERIC_DEVICES.items():
        results = analyze_atmospheric_claim(device)
        print_analysis(results)
        print()

    compare_to_alternatives()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Atmospheric electricity IS REAL
   - Earth-ionosphere system has ~300 kV potential
   - Fair-weather field is ~130 V/m at ground level
   - Global circuit carries ~1800 A total

2. It is NOT free energy
   - Energy comes from global thunderstorm circuit
   - ~2000 active thunderstorms maintain the field
   - Total power: ~400 kW globally

3. Practical extraction is extremely difficult
   - Current density: ~2 pA/m² (picoamps!)
   - A football field-sized collector at 100m height: ~0.1 microwatts
   - Solar panel same size: ~500,000 watts

4. Claims to watch for:
   - "Free energy" - No, it comes from thunderstorms
   - "Unlimited power" - No, global circuit is ~400 kW total
   - "Tesla proved it" - Tesla's actual experiments had modest results
   - "Over-unity" - Impossible, violates thermodynamics
""")


if __name__ == "__main__":
    main()
