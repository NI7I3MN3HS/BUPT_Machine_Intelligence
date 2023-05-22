function randomGenerate() {
  var digits = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  digits.sort(() => Math.random() - 0.5);

  var startInput = document.getElementById("start_state");
  var targetInput = document.getElementById("target_state");

  startInput.value = digits.join("");
  digits.sort(() => Math.random() - 0.5);
  targetInput.value = digits.join("");
}

function clearStates() {
  document.getElementById("start_state").value = "";
  document.getElementById("target_state").value = "";
}

function validateInput() {
  var startInput = document.getElementById("start_state");
  var targetInput = document.getElementById("target_state");
  var startValue = startInput.value;
  var targetValue = targetInput.value;
  var validPattern = /^[0-8]{9}$/;

  if (!validPattern.test(startValue)) {
    displayErrorMessage(startInput, "初始状态必须由0至8的9个数字组成");
    return false;
  }

  if (!validPattern.test(targetValue)) {
    displayErrorMessage(targetInput, "目标状态必须由0至8的9个数字组成");
    return false;
  }

  return true;
}

function displayErrorMessage(inputElement, message) {
  var errorElement = inputElement.parentNode.querySelector(".error-message");
  if (!errorElement) {
    errorElement = document.createElement("p");
    errorElement.className = "error-message";
    inputElement.parentNode.appendChild(errorElement);
  }
  errorElement.textContent = message;
}
