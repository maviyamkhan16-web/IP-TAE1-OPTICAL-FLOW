import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, amount=0.02):
    noisy = image.copy()

    num_salt = int(amount * image.shape[0] * image.shape[1] * 0.5)
    coords = tuple(
        np.random.randint(0, i, num_salt)
        for i in image.shape[:2]
    )

    if len(image.shape) == 2:
        noisy[coords] = 255
    else:
        noisy[coords[0], coords[1]] = 255

    num_pepper = int(amount * image.shape[0] * image.shape[1] * 0.5)
    coords = tuple(
        np.random.randint(0, i, num_pepper)
        for i in image.shape[:2]
    )

    if len(image.shape) == 2:
        noisy[coords] = 0
    else:
        noisy[coords[0], coords[1]] = 0

    return noisy


def apply_filter(image, filter_type="Gaussian", kernel_size=5):
    if filter_type == "Gaussian":
        return cv2.GaussianBlur(
            image,
            (kernel_size, kernel_size),
            0
        )
    elif filter_type == "Median":
        return cv2.medianBlur(image, kernel_size)
    elif filter_type == "Bilateral":
        return cv2.bilateralFilter(
            image,
            kernel_size,
            75,
            75
        )
    return image


def show_histogram(image, title):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    plt.figure(figsize=(8, 4))
    plt.hist(gray.ravel(), bins=256, range=[0, 256])
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])
    plt.tight_layout()
    plt.show()


def calculate_optical_flow(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    start_time = time.time()

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

    execution_time = time.time() - start_time

    return flow, execution_time


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

    return flow_image, magnitude, angle


def draw_flow_arrows(frame, flow, step=16):
    result = frame.copy()
    height, width = frame.shape[:2]

    for y in range(0, height, step):
        for x in range(0, width, step):
            fx, fy = flow[y, x]

            end_x = int(x + fx)
            end_y = int(y + fy)

            cv2.arrowedLine(
                result,
                (x, y),
                (end_x, end_y),
                (0, 255, 0),
                1,
                tipLength=0.3
            )

            cv2.circle(
                result,
                (x, y),
                1,
                (0, 0, 255),
                -1
            )

    return result


def calculate_flow_statistics(magnitude):
    mean_motion = np.mean(magnitude)
    max_motion = np.max(magnitude)

    moving_pixels = np.sum(magnitude > 1.0)
    total_pixels = magnitude.size

    motion_percentage = (
        moving_pixels / total_pixels
    ) * 100

    return (
        mean_motion,
        max_motion,
        motion_percentage
    )


def compare_frames(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    mse = np.mean(
        (
            gray1.astype(float) -
            gray2.astype(float)
        ) ** 2
    )

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10(
            (255 ** 2) / mse
        )

    return mse, psnr


def add_information(
    image,
    frame_number,
    mean_motion,
    max_motion,
    motion_percentage,
    execution_time
):
    result = image.copy()

    cv2.putText(
        result,
        f"Frame: {frame_number}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        result,
        f"Mean Motion: {mean_motion:.2f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        result,
        f"Max Motion: {max_motion:.2f}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        result,
        f"Moving Pixels: {motion_percentage:.2f}%",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        result,
        f"Execution Time: {execution_time:.4f}s",
        (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    return result


def main():
    video_path = (
        r"C:\Users\User\PycharmProjects"
        r"\Maviya_Mishkat\input_video.mp4"
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps <= 0:
        original_fps = 30

    delay = max(
        1,
        int(1000 / original_fps)
    )

    width = 640
    height = 480
    kernel_size = 5

    ret, frame1 = cap.read()

    if not ret:
        print("Error: Could not read first frame.")
        cap.release()
        return

    frame1 = cv2.resize(
        frame1,
        (width, height)
    )

    noisy_frame1 = add_gaussian_noise(frame1)

    filtered_frame1 = apply_filter(
        noisy_frame1,
        "Gaussian",
        kernel_size
    )

    show_histogram(
        frame1,
        "Histogram - Original Frame"
    )

    show_histogram(
        noisy_frame1,
        "Histogram - Noisy Frame"
    )

    show_histogram(
        filtered_frame1,
        "Histogram - Filtered Frame"
    )

    frame_number = 2
    total_processing_time = 0
    processed_frames = 0

    print("\n======================================")
    print("FARNEBACK OPTICAL FLOW STARTED")
    print("======================================")

    while True:
        ret, frame2 = cap.read()

        if not ret:
            break

        frame2 = cv2.resize(
            frame2,
            (width, height)
        )

        noisy_frame2 = add_gaussian_noise(frame2)

        filtered_frame2 = apply_filter(
            noisy_frame2,
            "Gaussian",
            kernel_size
        )

        flow, execution_time = calculate_optical_flow(
            filtered_frame1,
            filtered_frame2
        )

        total_processing_time += execution_time
        processed_frames += 1

        flow_image, magnitude, angle = visualize_flow(
            filtered_frame1,
            flow
        )

        arrow_image = draw_flow_arrows(
            filtered_frame1,
            flow
        )

        (
            mean_motion,
            max_motion,
            motion_percentage
        ) = calculate_flow_statistics(magnitude)

        mse, psnr = compare_frames(
            frame1,
            frame2
        )

        processing_fps = (
            1 / execution_time
            if execution_time > 0
            else 0
        )

        flow_display = add_information(
            flow_image,
            frame_number,
            mean_motion,
            max_motion,
            motion_percentage,
            execution_time
        )

        original_display = frame2.copy()

        cv2.putText(
            original_display,
            f"Frame: {frame_number}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        filtered_display = filtered_frame2.copy()

        cv2.putText(
            filtered_display,
            "Gaussian Filter",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        noisy_display = noisy_frame2.copy()

        cv2.putText(
            noisy_display,
            "Gaussian Noise",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Original Video",
            original_display
        )

        cv2.imshow(
            "Noisy Frame",
            noisy_display
        )

        cv2.imshow(
            "Filtered Frame",
            filtered_display
        )

        cv2.imshow(
            "Farneback Dense Optical Flow",
            flow_display
        )

        cv2.imshow(
            "Optical Flow Arrows",
            arrow_image
        )

        print(
            f"Frame {frame_number:4d} | "
            f"Time: {execution_time:.4f}s | "
            f"FPS: {processing_fps:.2f} | "
            f"Mean Motion: {mean_motion:.2f} | "
            f"Max Motion: {max_motion:.2f} | "
            f"Moving Pixels: {motion_percentage:.2f}% | "
            f"MSE: {mse:.2f} | "
            f"PSNR: {psnr:.2f} dB"
        )

        filtered_frame1 = filtered_frame2
        frame1 = frame2

        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('p'):
            cv2.waitKey(0)

        frame_number += 1

    if processed_frames > 0:
        average_execution_time = (
            total_processing_time /
            processed_frames
        )

        average_fps = (
            1 / average_execution_time
            if average_execution_time > 0
            else 0
        )
    else:
        average_execution_time = 0
        avrage_fps = 0
    cap.release()
    cv2.destroyAllWindows()
    print("\n======================================")
    print("FINAL RESULTS")
    print("======================================")
    print(
        f"Frames Processed       : {processed_frames}"
    )
    print(
        f"Average Execution Time : "
        f"{average_execution_time:.4f} seconds"
    )
    print(
        f"Average Processing FPS : {average_fps:.2f}"
    )
    print("Video processing completed.")
if __name__ == "__main__":
    main()