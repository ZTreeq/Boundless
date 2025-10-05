document.getElementById("fileInput").addEventListener("change", function() {
  const file = this.files[0];
  const uploadedImg = document.getElementById("uploadedImg");
  const annotatedImg = document.getElementById("annotatedImg");
  if (file) {
    uploadedImg.src = URL.createObjectURL(file);
    uploadedImg.style.display = "block";
    annotatedImg.style.display = "none";
    document.getElementById("output").innerText = "";
  }
});

document.getElementById("uploadBtn").addEventListener("click", async () => {
  let fileInput = document.getElementById("fileInput");
  let uploadedImg = document.getElementById("uploadedImg");
  let annotatedImg = document.getElementById("annotatedImg");
  let output = document.getElementById("output");

  if (!fileInput.files.length) {
    alert("Please select a file first");
    return;
  }

  let file = fileInput.files[0];
  let formData = new FormData();
  formData.append("file", file);

  output.innerText = "Processing...";
  annotatedImg.style.display = "none";

  try {
    let res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData
    });

    if (!res.ok) throw new Error("Server error");

    // Expecting backend to return { "annotated_image": "<base64 string>", ... }
    let data = await res.json();

    if (data.annotated_image) {
      annotatedImg.src = "data:image/png;base64," + data.annotated_image;
      annotatedImg.style.display = "block";
      output.innerText = "";
    } else {
      output.innerText = "No annotated image returned.";
    }
  } catch (err) {
    output.innerText = "Error: " + err;
  }
});
