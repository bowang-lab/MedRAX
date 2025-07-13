import cv2
import numpy as np
from medrax.tools.utils import BaseTool, tool
from medrax.utils.utils import get_image_path, get_image_from_path

class CTRTool(BaseTool):
    """
    A tool for calculating the cardiothoracic ratio (CTR) from a chest X-ray image.
    """

    @tool("cardiothoracic_ratio_tool")
    def _run(self, image_path: str) -> dict:
        """
        Calculates the cardiothoracic ratio (CTR) from a chest X-ray image.

        Args:
            image_path (str): The path to the chest X-ray image.

        Returns:
            dict: A dictionary containing the CTR value and the path to the visualized image.
        """
        try:
            image_path = get_image_path(image_path)
            image = get_image_from_path(image_path)

            # Placeholder for MedSAM integration
            # In a real implementation, MedSAM would be used to segment the heart and lungs
            # For now, we will use a simplified method based on image processing

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Threshold the image to get a binary mask
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Assume the largest contour is the thoracic cavity
            thoracic_contour = max(contours, key=cv2.contourArea)

            # Get the bounding box of the thoracic cavity
            x, y, w, h = cv2.boundingRect(thoracic_contour)
            thoracic_width = w

            # Assume the heart is the second largest contour
            # This is a major simplification and will be replaced by MedSAM
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            if len(contours) > 1:
                heart_contour = contours[1]
                x, y, w, h = cv2.boundingRect(heart_contour)
                cardiac_width = w
            else:
                # Fallback if only one contour is found
                cardiac_width = thoracic_width / 2

            # Calculate CTR
            ctr = cardiac_width / thoracic_width

            # Create a visualization
            output_image = image.copy()
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.line(output_image, (x, y + h // 2), (x + w, y + h // 2), (0, 255, 0), 2)
            cv2.putText(output_image, f"CTR: {ctr:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Save the visualized image
            output_path = get_image_path(image_path, suffix="_ctr.png")
            cv2.imwrite(output_path, output_image)

            return {"ctr_value": ctr, "image_path": output_path}

        except Exception as e:
            return {"error": str(e)}
