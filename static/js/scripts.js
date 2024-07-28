const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const fileForm = document.getElementById('file-form');

dropArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropArea.classList.add('dragging');
});

dropArea.addEventListener('dragleave', () => {
    dropArea.classList.remove('dragging');
});

dropArea.addEventListener('drop', (event) => {
    event.preventDefault();
    dropArea.classList.remove('dragging');
    console.log('dragged removed')
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        fileForm.submit();
        console.log('submitted file')
    }
});