import cv2
import numpy as np
import time


def add_gaussian_noise(image, mean=0, sigma=25):
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy_image = image.astype(np.float32) + gaussian
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, amount=0.02):
    noisy_image = image.copy()
    row, col, _ = image.shape

    num_salt = int(np.ceil(amount * image.size * 0.5))
    num_pepper = int(np.ceil(amount * image.size * 0.5))

    coords = [
        np.random.randint(0, row, num_salt),
        np.random.randint(0, col, num_salt)
    ]

    noisy_image[coords[0], coords[1]] = 255

    coords = [
        np.random.randint(0, row, num_pepper),
        np.random.randint(0, col, num_pepper)
    ]

    noisy_image[coords[0], coords[1]] = 0

    return noisy_image


def apply_filter(image, filter_type="Gaussian", kernel_size=5):

    if filter_type == "Gaussian":
        return cv2.GaussianBlur(
            image,
            (kernel_size, kernel_size),
            0
        )

    elif filter_type == "Median":
        return cv2.medianBlur(
            image,
            kernel_size
        )

    elif filter_type == "Bilateral":
        return cv2.bilateralFilter(
            image,
            9,
            75,
            75
        )

    return image


def calculate_optical_flow(frame1, frame2):

    gray1 = cv2.cvtColor(
        frame1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        frame2,
        cv2.COLOR_BGR2GRAY
    )

    flow = cv2.calcOpticalFlowFarneback(
        gray1,
        gray2,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    return flow


def visualize_flow(frame, flow):

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    hsv = np.zeros_like(frame)

    hsv[..., 1] = 255

    hsv[..., 0] = angle * 180 / np.pi / 2

    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    flow_image = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    return flow_image, magnitude


def draw_flow_arrows(frame, flow, step=16):

    output = frame.copy()

    height, width = frame.shape[:2]

    for y in range(0, height, step):

        for x in range(0, width, step):

            fx, fy = flow[y, x]

            end_x = int(x + fx)
            end_y = int(y + fy)

            cv2.arrowedLine(
                output,
                (x, y),
                (end_x, end_y),
                (0, 255, 0),
                1,
                tipLength=0.3
            )

    return output


def calculate_flow_statistics(magnitude):

    mean_motion = float(
        np.mean(magnitude)
    )

    max_motion = float(
        np.max(magnitude)
    )

    moving_pixels = np.sum(
        magnitude > 1.0
    )

    total_pixels = magnitude.size

    moving_pixel_percentage = (
        moving_pixels / total_pixels
    ) * 100

    return (
        mean_motion,
        max_motion,
        moving_pixel_percentage
    )


def compare_frames(frame1, frame2):

    gray1 = cv2.cvtColor(
        frame1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        frame2,
        cv2.COLOR_BGR2GRAY
    )

    mse = np.mean(
        (
            gray1.astype(np.float32)
            -
            gray2.astype(np.float32)
        ) ** 2
    )

    if mse == 0:

        psnr = float("inf")

    else:

        psnr = 10 * np.log10(
            (255 ** 2) / mse
        )

    return float(mse), float(psnr)


def process_video(
    input_path,
    output_path,
    noise_type="Gaussian",
    filter_type="Gaussian",
    kernel_size=5
):

    cap = cv2.VideoCapture(
        input_path
    )

    if not cap.isOpened():
        raise ValueError(
            "Unable to open input video"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30

    width = 640
    height = 480

    fourcc = cv2.VideoWriter_fourcc(
        *"avc1"
    )

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    ret, frame1 = cap.read()

    if not ret:

        cap.release()
        out.release()

        raise ValueError(
            "Unable to read video"
        )

    frame1 = cv2.resize(
        frame1,
        (width, height)
    )

    if noise_type == "Gaussian":

        noisy_frame1 = add_gaussian_noise(
            frame1
        )

    elif noise_type == "Salt & Pepper":

        noisy_frame1 = add_salt_pepper_noise(
            frame1
        )

    else:

        noisy_frame1 = frame1.copy()

    filtered_frame1 = apply_filter(
        noisy_frame1,
        filter_type,
        kernel_size
    )

    frame_count = 0
    total_processing_time = 0

    total_mean_motion = 0
    total_max_motion = 0
    total_moving_percentage = 0
    total_mse = 0
    total_psnr = 0

    valid_psnr_count = 0

    while True:

        ret, frame2 = cap.read()

        if not ret:
            break

        start_time = time.time()

        frame2 = cv2.resize(
            frame2,
            (width, height)
        )

        if noise_type == "Gaussian":

            noisy_frame2 = add_gaussian_noise(
                frame2
            )

        elif noise_type == "Salt & Pepper":

            noisy_frame2 = add_salt_pepper_noise(
                frame2
            )

        else:

            noisy_frame2 = frame2.copy()

        filtered_frame2 = apply_filter(
            noisy_frame2,
            filter_type,
            kernel_size
        )

        flow = calculate_optical_flow(
            filtered_frame1,
            filtered_frame2
        )

        flow_image, magnitude = visualize_flow(
            filtered_frame2,
            flow
        )

        arrow_image = draw_flow_arrows(
            filtered_frame2,
            flow
        )

        (
            mean_motion,
            max_motion,
            moving_percentage
        ) = calculate_flow_statistics(
            magnitude
        )

        mse, psnr = compare_frames(
            filtered_frame1,
            filtered_frame2
        )

        display_frame = flow_image.copy()

        cv2.putText(
            display_frame,
            f"Frame: {frame_count + 1}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Mean Motion: {mean_motion:.2f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display_frame,
            f"Max Motion: {max_motion:.2f}",
            (10, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display_frame,
            f"Moving Pixels: {moving_percentage:.2f}%",
            (10, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            display_frame,
            f"MSE: {mse:.2f}",
            (10, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        if np.isfinite(psnr):

            cv2.putText(
                display_frame,
                f"PSNR: {psnr:.2f} dB",
                (10, 138),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        combined = np.hstack(
            (
                display_frame,
                arrow_image
            )
        )

        combined = cv2.resize(
            combined,
            (width, height)
        )

        out.write(
            combined
        )

        processing_time = (
            time.time() - start_time
        )

        total_processing_time += (
            processing_time
        )

        total_mean_motion += (
            mean_motion
        )

        total_max_motion += (
            max_motion
        )

        total_moving_percentage += (
            moving_percentage
        )

        total_mse += mse

        if np.isfinite(psnr):

            total_psnr += psnr
            valid_psnr_count += 1

        frame_count += 1

        filtered_frame1 = filtered_frame2

    cap.release()
    out.release()

    if frame_count == 0:

        raise ValueError(
            "Video does not contain enough frames"
        )

    average_mean_motion = (
        total_mean_motion /
        frame_count
    )

    average_max_motion = (
        total_max_motion /
        frame_count
    )

    average_moving_percentage = (
        total_moving_percentage /
        frame_count
    )

    average_mse = (
        total_mse /
        frame_count
    )

    if valid_psnr_count > 0:

        average_psnr = (
            total_psnr /
            valid_psnr_count
        )

    else:

        average_psnr = float("inf")

    average_processing_time = (
        total_processing_time /
        frame_count
    )

    processing_fps = (
        1 /
        average_processing_time
        if average_processing_time > 0
        else 0
    )

    return {
        "frames_processed": frame_count,
        "video_fps": round(
            float(fps),
            2
        ),
        "processing_fps": round(
            float(processing_fps),
            2
        ),
        "average_processing_time": round(
            float(average_processing_time),
            4
        ),
        "mean_motion": round(
            float(average_mean_motion),
            4
        ),
        "max_motion": round(
            float(average_max_motion),
            4
        ),
        "moving_pixel_percentage": round(
            float(average_moving_percentage),
            2
        ),
        "mse": round(
            float(average_mse),
            4
        ),
        "psnr": (
            "Infinity"
            if not np.isfinite(average_psnr)
            else round(
                float(average_psnr),
                4
            )
        ),
        "noise_type": noise_type,
        "filter_type": filter_type,
        "kernel_size": kernel_size
    }