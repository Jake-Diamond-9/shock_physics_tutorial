# Shock Physics and the Plate Impact Experiment: A Tutorial

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Jake-Diamond-9/shock_physics_tutorial/HEAD?urlpath=%2Fdoc%2Ftree%2Fshock_physics_tutorial.ipynb)

**Associated Paper:** *"Shock Physics and the Plate Impact Experiment: A Tutorial"*  
**Paper DOI:** TBD

---

## Run Online (No Installation Required)

Click the **launch binder** badge above to open and execute `shock_physics_tutorial.ipynb` directly in your web browser. 

> **Note:** Binder sessions close after 10 minutes of inactivity. If you modify the notebook and wish to keep your changes, go to **File -> Download** before closing your browser tab.

---

## Usage Examples

Import the module and call `shock_physics_tutorial()` using one of the three supported calculation modes:

```python
from shock_physics_tutorial import *

# 1. IMPACT MODE
proj, targ = shock_physics_tutorial(
    Mode="Impact", 
    Velocity=3000, 
    ProjName="Ta", 
    ProjThick=1e-3, 
    TargName="Cu", 
    TargThick=5e-3, 
    Verbose=True
)

# 2. PRESSURE MODE
targ = shock_physics_tutorial(
    Mode="Pressure", 
    Pressure=100e9, 
    Duration=0.5e-6, 
    TargName="Al 2024", 
    TargThick=5e-3, 
    Verbose=True
)

# 3. VOLUME MODE
targ = shock_physics_tutorial(
    Mode="Volume", 
    VolumeRatio=0.85, 
    Duration=0.5e-6, 
    TargName="Mo", 
    TargThick=5e-3, 
    Verbose=True
)
