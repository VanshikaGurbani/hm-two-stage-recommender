import os

# Tests must never touch the real (large, licensed) data or a competition zip lying around.
os.environ["HM_FORCE_SYNTHETIC"] = "1"
