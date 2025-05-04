function startVoiceAssistant() {
  // Logic to start the voice assistant and handle voice input
  alert("Voice assistant activated. Start speaking...");

  // Example of integrating Web Speech API (if you plan to use it)
  if ('webkitSpeechRecognition' in window) {
      let recognition = new webkitSpeechRecognition();
      recognition.lang = "en-US";
      recognition.start();

      recognition.onresult = function(event) {
          document.getElementById('userQuery').value = event.results[0][0].transcript;
      };

      recognition.onerror = function(event) {
          console.error("Speech recognition error: ", event.error);
      };

      recognition.onend = function() {
          console.log("Speech recognition ended.");
      };
  } else {
      alert("Your browser does not support voice input.");
  }
}
