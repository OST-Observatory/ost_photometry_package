"""Image type utilities for data reduction."""

import numpy as np
import ccdproc as ccdp


def get_image_type(
    image_file_collection: ccdp.ImageFileCollection,
    image_type_dict: dict[str, list[str]] | list[str],
    image_class: str | None = None,
) -> str | list[str] | None:
    """
    From an image file collection get the existing image type from a
    list of possible images

    Parameters
    ----------
    image_file_collection
        Image file collection

    image_type_dict
        Image types of the images.
        Possibilities: bias, dark, flat, light

    image_class
        Image file type class to look for.
        Default is ``None``.

    Returns
    -------
    image_types
        Image types or list of image types
    """
    #   Create mask
    if not image_class:
        mask = [
            True if image_type in image_file_collection.summary["imagetyp"] else False
            for image_type in image_type_dict
        ]
    else:
        mask = [
            True if image_type in image_file_collection.summary["imagetyp"] else False
            for image_type in image_type_dict[image_class]
        ]

    #   Get image type ID
    id_image_type = np.argwhere(mask).ravel()
    if not id_image_type.size:
        return None

    #   Get image type
    #   Restricted to only one result -> this is currently necessary
    id_image_type = id_image_type[0]

    #   Return the image type
    if not image_class:
        return image_type_dict[id_image_type]
    else:
        return image_type_dict[image_class][id_image_type]
