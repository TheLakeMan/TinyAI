// DOM Elements
const promptInput = document.getElementById('prompt-input');
const generateBtn = document.getElementById('generate-btn');
const clearBtn = document.getElementById('clear-btn');
const resultContainer = document.getElementById('result-container');
const initialMessage = document.getElementById('initial-message');
const resultText = document.getElementById('result-text');
const loadingOverlay = document.getElementById('loading-overlay');

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Setup event listeners
    generateBtn.addEventListener('click', handleGenerate);
    clearBtn.addEventListener('click', handleClear);

    // Initialize
    promptInput.focus();
});

// Handle Generate Button Click
async function handleGenerate() {
    const prompt = promptInput.value.trim();

    if (!prompt) {
        showError('Please enter a prompt first.');
        promptInput.focus();
        return;
    }

    try {
        // Show loading state
        showLoading(true);

        // Make API request to local server
        const result = await generateText(prompt);

        // Display result
        showResult(result);
    } catch (error) {
        showError(`Error: ${error.message || 'Failed to generate text'}`);
    } finally {
        // Hide loading state
        showLoading(false);
    }
}

// Handle Clear Button Click
function handleClear() {
    // Clear prompt input
    promptInput.value = '';

    // Reset result area
    resultText.textContent = '';
    initialMessage.style.display = 'block';

    // Remove any error messages
    const errorElement = resultContainer.querySelector('.error-message');
    if (errorElement) {
        errorElement.remove();
    }

    // Focus the prompt input
    promptInput.focus();
}

// API Request Function - Mock implementation for demonstration
async function generateText(prompt) {
    // We're simulating a delay to mimic a network request
    await new Promise(resolve => setTimeout(resolve, 1500));

    // Simple mock response based on the prompt
    if (!prompt || prompt.trim().length === 0) {
        throw new Error('Empty prompt provided');
    }

    // Create a basic mock response
    const responses = [
        `TinyAI is processing: "${prompt}"\n\nThis is a mock response since the actual TinyAI model backend is not running. In a real implementation, this would connect to the C-based web server that loads the TinyAI model and tokenizer.`,
        `Response to: "${prompt}"\n\nThis is a demonstration of the TinyAI web interface. The actual text generation would require compiling and running the backend server implemented in interface/web_server.c with all its dependencies.`,
        `Prompt: "${prompt}"\n\nTinyAI is an ultra-lightweight AI model designed for efficient inference on resource-constrained devices. This is a mock response for demonstration purposes.`
    ];

    // Return a random response from the array
    return responses[Math.floor(Math.random() * responses.length)];
}

// Show Result Function
function showResult(text) {
    // Hide initial message
    initialMessage.style.display = 'none';

    // Set result text
    resultText.textContent = text;

    // Remove any error messages
    const errorElement = resultContainer.querySelector('.error-message');
    if (errorElement) {
        errorElement.remove();
    }
}

// Show Error Function
function showError(message) {
    // Check if error element already exists
    let errorElement = resultContainer.querySelector('.error-message');

    if (!errorElement) {
        // Create error element if it doesn't exist
        errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        resultContainer.appendChild(errorElement);
    }

    // Set error message
    errorElement.textContent = message;
}

// Control Loading Overlay
function showLoading(show) {
    if (show) {
        loadingOverlay.classList.remove('hidden');
        generateBtn.disabled = true;
    } else {
        loadingOverlay.classList.add('hidden');
        generateBtn.disabled = false;
    }
}

// Keyboard Shortcuts
document.addEventListener('keydown', (event) => {
    // Ctrl+Enter to generate
    if (event.ctrlKey && event.key === 'Enter') {
        if (document.activeElement === promptInput) {
            handleGenerate();
        }
    }

    // Escape to clear
    if (event.key === 'Escape') {
        handleClear();
    }
});
