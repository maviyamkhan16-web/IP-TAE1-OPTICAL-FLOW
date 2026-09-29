const videoInput = document.getElementById("videoInput");
const dropZone = document.getElementById("dropZone");

const selectedFile = document.getElementById("selectedFile");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");

const removeFile = document.getElementById("removeFile");

const processButton = document.getElementById("processButton");
const resetButton = document.getElementById("resetButton");

const buttonText = document.getElementById("buttonText");
const statusText = document.getElementById("status");

const progressContainer =
    document.getElementById("progressContainer");

const resultSection =
    document.getElementById("resultSection");

const resultVideo =
    document.getElementById("resultVideo");


dropZone.addEventListener("click", () => {
    videoInput.click();
});


videoInput.addEventListener("change", () => {

    if (videoInput.files.length > 0) {
        showSelectedFile(videoInput.files[0]);
    }

});


dropZone.addEventListener("dragover", (event) => {

    event.preventDefault();

    dropZone.classList.add("dragover");

});


dropZone.addEventListener("dragleave", () => {

    dropZone.classList.remove("dragover");

});


dropZone.addEventListener("drop", (event) => {

    event.preventDefault();

    dropZone.classList.remove("dragover");

    const files = event.dataTransfer.files;

    if (files.length > 0) {

        const file = files[0];

        if (!file.type.startsWith("video/")) {

            statusText.textContent =
                "Please select a valid video file.";

            return;
        }

        videoInput.files = files;

        showSelectedFile(file);
    }

});


function showSelectedFile(file) {

    fileName.textContent = file.name;

    fileSize.textContent =
        formatFileSize(file.size);

    selectedFile.style.display = "flex";

    statusText.textContent =
        "Video ready for analysis.";

}


function formatFileSize(bytes) {

    if (bytes < 1024 * 1024) {

        return (
            (bytes / 1024).toFixed(1)
            + " KB"
        );

    }

    return (
        (bytes / (1024 * 1024)).toFixed(2)
        + " MB"
    );

}


removeFile.addEventListener("click", () => {

    videoInput.value = "";

    selectedFile.style.display = "none";

    statusText.textContent =
        "";

});


resetButton.addEventListener("click", () => {

    videoInput.value = "";

    selectedFile.style.display = "none";

    resultSection.classList.add("hidden");

    resultVideo.removeAttribute("src");

    resultVideo.load();

    statusText.textContent = "";

    progressContainer.style.display = "none";

});


processButton.addEventListener("click", async () => {

    if (!videoInput.files.length) {

        statusText.textContent =
            "Please select a video first.";

        return;
    }


    const video = videoInput.files[0];

    const noiseType =
        document.getElementById(
            "noiseType"
        ).value;

    const filterType =
        document.getElementById(
            "filterType"
        ).value;

    const kernelSize =
        document.getElementById(
            "kernelSize"
        ).value;


    const formData = new FormData();

    formData.append(
        "video",
        video
    );

    formData.append(
        "noise_type",
        noiseType
    );

    formData.append(
        "filter_type",
        filterType
    );

    formData.append(
        "kernel_size",
        kernelSize
    );


    processButton.disabled = true;

    buttonText.textContent =
        "Processing...";

    progressContainer.style.display =
        "block";

    statusText.textContent =
        "Farneback optical flow is analyzing your video.";

    resultSection.classList.add("hidden");


    try {

        const response = await fetch(
            "http://127.0.0.1:5000/process",
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.error ||
                "Video processing failed."
            );

        }


        resultVideo.src =
            "http://127.0.0.1:5000/output/" +
            data.output_video;


        resultVideo.load();


        const stats =
            data.statistics;


        animateValue(
            "frames",
            stats.frames_processed
        );

        animateValue(
            "videoFps",
            stats.video_fps
        );

        animateValue(
            "processingFps",
            stats.processing_fps
        );

        animateValue(
            "meanMotion",
            stats.mean_motion
        );

        animateValue(
            "maxMotion",
            stats.max_motion
        );

        animateValue(
            "movingPixels",
            stats.moving_pixel_percentage +
            "%"
        );

        animateValue(
            "mse",
            stats.mse
        );

        animateValue(
            "psnr",
            stats.psnr
        );


        resultSection.classList.remove(
            "hidden"
        );


        statusText.textContent =
            "Analysis completed successfully.";

        progressContainer.style.display =
            "none";


        resultSection.scrollIntoView({
            behavior: "smooth"
        });


    } catch (error) {

        console.error(error);

        statusText.textContent =
            "Error: " + error.message;

        progressContainer.style.display =
            "none";

    } finally {

        processButton.disabled = false;

        buttonText.textContent =
            "Analyze Video";

    }

});


function animateValue(
    elementId,
    value
) {

    const element =
        document.getElementById(
            elementId
        );

    element.textContent = value;

}