function randomGenerate() {
  var digits = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  digits.sort(() => Math.random() - 0.5);

  var startInput = document.getElementById("start_state");
  var targetInput = document.getElementById("target_state");

  startInput.value = digits.join("");
  targetInput.value = "012345678";
}
