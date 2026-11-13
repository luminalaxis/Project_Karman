"""fsk_lapsim — quasi-steady-state lap time simulator and aero sensitivity tool for FSK EVs.

Units are SI throughout (m, s, kg, N, W, rad). See MODEL.md for the equations.
"""

from .aero import Aero, AeroComponent
from .powertrain import Powertrain
from .tire import Tire
from .vehicle import Vehicle, load_vehicle, with_param

__all__ = ["Aero", "AeroComponent", "Powertrain", "Tire", "Vehicle", "load_vehicle", "with_param"]
__version__ = "0.1.0"
