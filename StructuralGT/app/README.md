# StructuralGT GUI

Automated graph-theory analysis of structural network images.

---

## Installation

#### Option 1: From source

To run the application from source:
```bash
# Create environment
micromamba env create -f environment.yml
micromamba activate StructuralGT_GUI

# Run the application
python src/main.py
```

To build a standalone package for distribution (macOS only):
```bash
# Build the application
pyinstaller StructuralGT.spec

# Build DMG (requires create-dmg)
brew install create-dmg
bash build_dmg.sh
```

#### Option 2: Pre-built executable

Standalone executables for macOS and Windows are available in the [Dropbox download folder](https://www.dropbox.com/scl/fo/mph2fyj2qlvb5bam0uuy7/ABT_-Ho9zgq4Lu8zshFAwTk?rlkey=76efbxpwwhrnz3xpu01x60dpk&st=x5x3x6yu&dl=0).

---

## Getting started

The following steps describe the basic workflow of the application.

#### Load images/networks

The application supports three types of input:

| Type | Description |
|------|-------------|
| 2D Image | A folder containing a single image file |
| 3D Image | A folder containing a sequence of image files |
| Point Network | A CSV file with point coordinates |

---

#### Binarize the image

- Go to **Analysis -> Binarize Filter** on the side panel.
- Select binarize options. See below for detailed descriptions of the options.
- View the binarized image in the main window by selecting **Binarized Image** in the ribbon.

#### Extract the graph

- Go to **Analysis -> Graph Extraction** on the side panel.
- Select the weight type. See below for detailed descriptions of the options.
- Click **Extract Graph** to extract the graph.
- View the extracted graph in the main window by selecting **Extracted Graph** in the ribbon.

---

#### Compute GT parameters

- Go to **Analysis -> Graph Properties** on the side panel.
- Select the properties to compute. See below for detailed descriptions of the options.
- Click **Compute** to compute the properties.
- View the computed properties in the **Properties** on the side panel.
