─────────────────────────────
Layer Bottom Elevation Definitions for the 7-Model-Layer Setup
─────────────────────────────

Layer 1: Mixed Sand and Clay (MSC) (WSS Upper)

Source TIFFs in descending priority:
gm_numerical__08_upper_mid_sand_younger.tif (highest priority)
gm_numerical__07_upper_mid_clay_younger.tif
gm_numerical__06_middle_sand_younger.tif
gm_numerical__05_middle_clay_younger.tif
gm_numerical__04_lower_sand_younger.tif
gm_numerical__03_lower_clay_younger.tif
gm_numerical__02_basal_sand_younger.tif
gm_numerical__01_basal_clay_younger.tif
gm_numerical__00_bedrock_younger.tif (lowest priority)
Description:

To get a continuous surface, all of these surfaces are combined giving higher priority to the younger surfaces
─────────────────────────────

Layer 2: Upper Middle Clay (UMC)

Source TIFF:
gm_numerical__06_middle_sand_younger.tif
Description:
If that value is missing (NaN), the code is set up to default to the bottom of the WSS surface (see Layer 3).
─────────────────────────────

Layer 3: Wahpeton Shallow Sand + Wahpeton Sand Plain (WSS Lower/WSP)

Source TIFFs in descending priority:
gm_numerical__05_middle_clay_younger.tif (highest priority)
gm_numerical__04_lower_sand_younger.tif
gm_numerical__03_lower_clay_younger.tif
gm_numerical__02_basal_sand_younger.tif
gm_numerical__01_basal_clay_younger.tif
gm_numerical__00_bedrock_younger.tif (lowest priority)
To get a continuous surface, all of these surfaces are combined giving higher priority to the younger surfaces
─────────────────────────────

Layer 4: Lower Middle Clay (LMC)

Source TIFF:
gm_numerical__04_lower_sand_younger.tif
Description:
If a value is not available, the code will use the value from the WBV Bottom (Layer 5) as a fallback.
─────────────────────────────

Layer 5: Wahpeton Buried Valley (WBV)

Source TIFFs (from the WBV Bottom dictionary) in descending priority:
gm_numerical__03_lower_clay_younger.tif (highest priority)
gm_numerical__02_basal_sand_younger.tif
gm_numerical__01_basal_clay_younger.tif
gm_numerical__00_bedrock_younger.tif (lowest priority)
Description:
The WBV Bottom elevation is determined by iterating through these sources (in reverse-sorted order by their assigned numeric value). The first valid (non-NaN) value found becomes the bottom of the buried valley.
─────────────────────────────

Layer 6: Deep Clay (DC)

Source TIFF:
gm_numerical__02_basal_sand_younger.tif
Description:
This layer uses the WRV surface directly from the specified TIFF. If the WRV value is missing, the code replaces NaNs with the bedrock elevation from the bedrock surface (Layer 7).
─────────────────────────────

Layer 7: Wild Rice (WR)

Source TIFFs:
gm_numerical__00_bedrock_younger.tif (priority 0)
gm_numerical__01_basal_clay_younger.tif (priority 1)
Description:
For the bedrock surface, the code compares the two sources. The rule is: if the elevation from “bedrock” is greater than that from “bedrock_clay” or if “bedrock_clay” is missing, then the bedrock value is used; otherwise, the basal clay value is used. This final value represents the top of the bedrock and, by design, the bottom of the model.
─────────────────────────────
