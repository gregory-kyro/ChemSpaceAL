ATOMS = {
    "Ag",
    "Al",
    "Am",
    "Ar",
    "At",
    "Au",
    "D",
    "E",
    "Fe",
    "G",
    "K",
    "L",
    "M",
    "Ra",
    "Re",
    "Rf",
    "Rg",
    "Rh",
    "Ru",
    "T",
    "U",
    "V",
    "W",
    "Xe",
    "Y",
    "Zr",
    "a",
    "d",
    "f",
    "g",
    "h",
    "k",
    "m",
    "si",
    "t",
    "te",
    "u",
    "v",
    "y",
}


exceptions = set()
for atom in ATOMS:
    exceptions.add(atom)
    exceptions.add(f"[{atom}]")
import yaml

# dump set into a yaml file
with open("guacamole_exceptions.yaml", "w") as f:
    yaml.dump(list(exceptions), f)

print(len(ATOMS))
