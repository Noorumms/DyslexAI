import cv2
from pathlib import Path


# ============================================================
# NO FOLDER
# ============================================================

INPUT_FOLDER = Path(
    r"E:\FYP_RESEEARCH\DYSLEXAI_Code\DyslexAi-Handwriting-Module\Complete And Balance Hand-written Dataset\Dataset\Yes"
)

OUTPUT_FOLDER = Path(
    r"E:\FYP_RESEEARCH\DYSLEXAI_Code\DyslexAi-Handwriting-Module\Complete And Balance Hand-written Dataset\Dataset_Cropped\Yes"
    
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# SUPPORTED IMAGE TYPES
# ============================================================

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# ============================================================
# GET IMAGES
# ============================================================

images = sorted([
    file for file in INPUT_FOLDER.iterdir()
    if file.is_file()
    and file.suffix.lower() in EXTENSIONS
])


print("=" * 60)
print("DYSLEXAI - NO IMAGE CROPPING")
print("=" * 60)

print(f"Images found: {len(images)}")
print(f"Output folder: {OUTPUT_FOLDER}")

print()
print("CONTROLS")
print("Drag mouse = select written area")
print("ENTER / SPACE = confirm crop")
print("ESC = skip image")
print("Q = quit")
print("=" * 60)


# ============================================================
# PROCESS IMAGES
# ============================================================

for index, image_path in enumerate(images, start=1):

    output_path = OUTPUT_FOLDER / image_path.name

    # If already cropped, skip it
    if output_path.exists():
        print(
            f"[{index}/{len(images)}] "
            f"Already done: {image_path.name}"
        )
        continue


    # --------------------------------------------------------
    # LOAD ORIGINAL IMAGE
    # --------------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"ERROR: Could not open {image_path.name}")
        continue


    original_height, original_width = image.shape[:2]

    print()
    print(
        f"[{index}/{len(images)}] "
        f"{image_path.name}"
    )

    print(
        f"Original size: "
        f"{original_width} x {original_height}"
    )


    # --------------------------------------------------------
    # RESIZE ONLY FOR DISPLAY
    # --------------------------------------------------------

    max_width = 900
    max_height = 700

    scale = min(
        max_width / original_width,
        max_height / original_height,
        1.0
    )

    display_width = int(original_width * scale)
    display_height = int(original_height * scale)

    display_image = cv2.resize(
        image,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA
    )


    # --------------------------------------------------------
    # SELECT ROI
    # --------------------------------------------------------

    window_name = (
        f"NO | {index}/{len(images)} | "
        f"{image_path.name}"
    )

    roi = cv2.selectROI(
        window_name,
        display_image,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyAllWindows()


    # --------------------------------------------------------
    # GET ROI
    # --------------------------------------------------------

    x, y, width, height = roi


    # Nothing selected
    if width == 0 or height == 0:
        print("SKIPPED")
        continue


    # --------------------------------------------------------
    # CONVERT DISPLAY COORDINATES
    # TO ORIGINAL IMAGE COORDINATES
    # --------------------------------------------------------

    x = int(x / scale)
    y = int(y / scale)

    width = int(width / scale)
    height = int(height / scale)


    # Keep coordinates inside image
    x = max(0, x)
    y = max(0, y)

    width = min(
        width,
        original_width - x
    )

    height = min(
        height,
        original_height - y
    )


    # --------------------------------------------------------
    # CROP ORIGINAL IMAGE
    # --------------------------------------------------------

    cropped = image[
        y:y + height,
        x:x + width
    ]


    # --------------------------------------------------------
    # SAVE WITH SAME NAME
    # --------------------------------------------------------

    success = cv2.imwrite(
        str(output_path),
        cropped
    )


    if success:

        print(
            f"SAVED: {image_path.name}"
        )

        print(
            f"Crop size: {width} x {height}"
        )

    else:

        print(
            f"ERROR: Could not save "
            f"{image_path.name}"
        )


print()
print("=" * 60)
print("NO FOLDER COMPLETE")
print("=" * 60)