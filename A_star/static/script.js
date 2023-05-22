function randomGenerate() {
  var digits = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  digits.sort(() => Math.random() - 0.5);

  var startInputs = document
    .getElementById("start_state")
    .getElementsByTagName("input");

  for (let i = 0; i < startInputs.length; i++) {
    startInputs[i].value = digits[i];
  }

  digits.sort(() => Math.random() - 0.5);
  var targetInputs = document
    .getElementById("target_state")
    .getElementsByTagName("input");
  for (let i = 0; i < targetInputs.length; i++) {
    targetInputs[i].value = digits[i];
  }
}

function clearStates() {
  var startInputs = document
    .getElementById("start_state")
    .getElementsByTagName("input");
  for (let input of startInputs) {
    input.value = "";
  }
  var targetInputs = document
    .getElementById("target_state")
    .getElementsByTagName("input");
  for (let input of targetInputs) {
    input.value = "";
  }
}

function validateInput() {
  var startInputs = document
    .getElementById("start_state")
    .getElementsByTagName("input");
  var startValue = Array.from(startInputs)
    .map((input) => input.value)
    .join("");
  var targetInputs = document
    .getElementById("target_state")
    .getElementsByTagName("input");
  var targetValue = Array.from(targetInputs)
    .map((input) => input.value)
    .join("");
  var validPattern = /^(?!.*(.).*\1)[0-8]{9}$/;

  if (!validPattern.test(startValue)) {
    displayErrorMessage("start_state", "初始状态必须由0至8的9个不重复数字组成");
    return false;
  }

  if (!validPattern.test(targetValue)) {
    displayErrorMessage(
      "target_state",
      "目标状态必须由0至8的9个不重复数字组成"
    );
    return false;
  }

  return true;
}

function displayErrorMessage(stateId, message) {
  var errorElement = document.getElementById(stateId + "_error");
  errorElement.textContent = message;
}

function prepareAndValidateInput() {
  var startInputs = document
    .getElementById("start_state")
    .getElementsByTagName("input");
  var startValue = Array.from(startInputs)
    .map((input) => input.value)
    .join("");
  document.getElementById("start_state_string").value = startValue;

  var targetInputs = document
    .getElementById("target_state")
    .getElementsByTagName("input");
  var targetValue = Array.from(targetInputs)
    .map((input) => input.value)
    .join("");
  document.getElementById("target_state_string").value = targetValue;

  sessionStorage.setItem("start_state", collectInputs("start_state"));
  sessionStorage.setItem("target_state", collectInputs("target_state"));

  return validateInput();
}

// 在页面加载完成后，从 sessionStorage 中恢复用户的输入。
window.onload = function () {
  restoreInputs("start_state", sessionStorage.getItem("start_state"));
  restoreInputs("target_state", sessionStorage.getItem("target_state"));
};

// 收集一个表格的输入。
function collectInputs(tableId) {
  var inputs = document.getElementById(tableId).getElementsByTagName("input");
  var values = Array.from(inputs).map((input) => input.value);
  return values.join(",");
}

// 将一个表格的输入恢复到之前的状态。
function restoreInputs(tableId, valuesString) {
  var inputs = document.getElementById(tableId).getElementsByTagName("input");
  var values = valuesString.split(",");
  for (var i = 0; i < inputs.length; i++) {
    inputs[i].value = values[i];
  }
}
