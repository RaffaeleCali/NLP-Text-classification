async function classifyText() {
    const textInput = document.getElementById('textInput').value;
    const resultElement = document.getElementById('result');

    try {
        const response = await fetch('/classify', { 
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({text: textInput}),
        });

        if (!response.ok) {
            throw new Error(`Request failed with status ${response.status}`);
        }

        const data = await response.json();
        resultElement.textContent = ` ${data.class}`;
    } catch (error) {
        console.error("Errore nella classificazione del testo:", error);
        resultElement.textContent = "Errore nella richiesta di classificazione.";
    }
}
