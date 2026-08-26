console.log("Semantic word game!!!!!") // very important line

import "./style.css";
import { pipeline } from "@huggingface/transformers";
import { words } from 'popular-english-words';

// ooh magic numbers!
const MIN_SIMILARITY = 0.15;
const MAX_SIMILARITY = 0.85;

const MIN_FONT_SIZE = 16;
const MAX_FONT_SIZE = 64;

const WORD_PADDING_FACTOR = 5;

// Word data
const response = await fetch("./words.json");
const wordThemes = await response.json();

// Valid words
const validWords = new Set(words.getAll());
function isValidWord(word) {
    return validWords.has(word)
};

// Game state
let themeHue;

let secretWord;

let wordCloud;
let cloudWords = [];

let guessForm;
let wordInput;

let winMessage;
let winText;

let timer;
let startTime;
let timerInterval;
let pausedElapsedTime = 0;

let isPaused = false;
let gameOver = false;
let completedModes = new Set();
let currentMode;

// Store completed games so they can be reopened during the session.
const completedGames = new Map();

let guesses = new Set();

// important maths/data/library stuff
function cosineSimilarity(a, b) {
    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;
    for (let i = 0; i < a.data.length; i++) {
        dotProduct += a.data[i] * b.data[i];
        magnitudeA += a.data[i] * a.data[i];
        magnitudeB += a.data[i] * b.data[i];
    }
    magnitudeA = Math.sqrt(magnitudeA);
    magnitudeB = Math.sqrt(magnitudeB);
    console.log('Similarity is', dotProduct / (magnitudeA * magnitudeB));
    return dotProduct / (magnitudeA * magnitudeB);
};

function getElapsedMilliseconds() {
    return Date.now() - startTime;
}

function updateTimer() {
    const elapsedSeconds = Math.floor(getElapsedMilliseconds() / 1000);
    const minutes = Math.floor(elapsedSeconds / 60);
    const seconds = elapsedSeconds % 60;

    timer.textContent =
        `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function similarityToFontSize(similarity) {
    if (similarity < MIN_SIMILARITY) {return MIN_FONT_SIZE;}
    if (similarity > MAX_SIMILARITY) {return MAX_FONT_SIZE;}
    const x = (similarity - MIN_SIMILARITY) / (MAX_SIMILARITY - MIN_SIMILARITY);
    const curved = 3 * x ** 2 - 2 * x ** 3;
    return (MIN_FONT_SIZE + curved * (MAX_FONT_SIZE - MIN_FONT_SIZE));
};

const extractor = await pipeline("feature-extraction", "Xenova/paraphrase-MiniLM-L3-v2", {dtype: "q4"});
const embeddingCache = new Map();
async function getEmbedding(word) {
    if (embeddingCache.has(word)) {
        return embeddingCache.get(word);
    }
    const output = await extractor(word, {pooling: "mean", normalize: true});
    embeddingCache.set(word, output);
    return output;
};


// html stuff
const app = document.querySelector("#app");

showMenu();

// Game mode colours
const modeHues = {
    easy: 120,
    medium: 45,
    hard: 0
};

function showMenu() {
    clearInterval(timerInterval);

    app.innerHTML = `
        <main class="home">

            <header class="header">
                <h1>Semantic Word Game</h1>
                <p>Find the secret word.</p>
            </header>

            <div class="mode-selection">
                <button class="mode-button easy" data-mode="easy">[easy]</button>
                <button class="mode-button medium" data-mode="medium">[medium]</button>
                <button class="mode-button hard" data-mode="hard">[hard]</button>
            </div>

        </main>
    `;

    document.querySelectorAll(".mode-button").forEach(button => {
        const mode = button.dataset.mode;

        if (completedModes.has(mode)) {
            button.classList.add("complete");
        }

        button.addEventListener("click", () => {
            startGame(mode);
        });
    });
}

// physics stuff
class CloudWord {
    constructor(text, fontSize, similarity, savedX = null, savedY = null) {
        this.text = text;
        this.fontSize = fontSize;
        this.similarity = similarity;

        // Position
        const centreX = wordCloud.clientWidth / 2;
        const centreY = wordCloud.clientHeight / 2;
        const maximumRadius = 250;
        const similarityRatio = Math.max(0, Math.min(1, similarity / MAX_SIMILARITY));
        const spawnRadius = maximumRadius * (1 - similarityRatio);
        const angle = Math.random() * Math.PI * 2;
        const distance = Math.sqrt(Math.random()) * spawnRadius;

        if (savedX !== null && savedY !== null) {
            this.x = savedX;
            this.y = savedY;
        } else {
            this.x = centreX + Math.cos(angle) * distance;
            this.y = centreY + Math.sin(angle) * distance;
        }

        // Velocity
        this.vx = 0;
        this.vy = 0;

        // Physical properties
        this.mass = Math.max(1, fontSize / 16);
        this.element = document.createElement("span");
        this.element.textContent = text;
        this.element.classList.add("cloud-word");
        this.element.style.fontSize = `${fontSize}px`;
        this.element.style.setProperty("--word-colour", `hsl(${themeHue}, 65%, ${30 + Math.random() * 35}%)`);

        if (similarity >= 0.9999) {
            this.element.classList.add("correct");
        }

        wordCloud.appendChild(this.element);

        this.width = this.element.offsetWidth + WORD_PADDING_FACTOR;
        this.height = this.element.offsetHeight + WORD_PADDING_FACTOR;
    }


    update() {
        const centreX = wordCloud.clientWidth / 2;
        const centreY = wordCloud.clientHeight / 2;
        let forceX = 0;
        let forceY = 0;

        // Attraction towards centre
        const halfWidth = this.width / 2;
        const halfHeight = this.height / 2;
        const closestX = Math.max(this.x - halfWidth, Math.min(centreX, this.x + halfWidth));
        const closestY = Math.max(this.y - halfHeight, Math.min(centreY, this.y + halfHeight));
        let centreDX = centreX - closestX;
        let centreDY = centreY - closestY;
        const centreDistance = Math.sqrt(centreDX * centreDX + centreDY * centreDY);

        // If the centre is inside the word, fall back to using the word's centre.
        if (centreDistance === 0) {
            centreDX = centreX - this.x;
            centreDY = centreY - this.y;
        }

        const distance = Math.sqrt(centreDX * centreDX + centreDY * centreDY);

        if (distance > 0) {
            const attraction = 0.001 + this.similarity * 0.004;
            forceX += (centreDX / distance) * distance * attraction;
            forceY += (centreDY / distance) * distance * attraction;
        }

        // Repulsion from other words
        for (const other of cloudWords) {
            if (other === this) {continue;}
            const dx = this.x - other.x;
            const dy = this.y - other.y;
            const overlapX = (this.width + other.width) / 2 - Math.abs(dx);
            const overlapY = (this.height + other.height) / 2 - Math.abs(dy);
            if (overlapX <= 0 || overlapY <= 0) {continue;}

            // Which direction requires the least movement to resolve the collision?
            if (overlapX < overlapY * 1.3) {
                const direction = dx >= 0 ? 1 : -1;
                const push = overlapX * 0.25;
                forceX += direction * push * (other.mass / this.mass);

            } else {
                const direction = dy >= 0 ? 1 : -1;
                const push = overlapY * 0.25;
                forceY += direction * push * (other.mass / this.mass);
            }

            // Small sideways force allows words to slide around each other rather than forming vertical stacks
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance > 0) {
                const tangentX = -dy / distance;
                const tangentY = dx / distance;
                const slide = Math.min(overlapX, overlapY) * 0.08;

                forceX += tangentX * slide * (other.mass / this.mass);
                forceY += tangentY * slide * (other.mass / this.mass);
            }
        }


        // Convert force into acceleration
        this.vx += forceX / this.mass;
        this.vy += forceY / this.mass;

        // Damping
        this.vx *= 0.82;
        this.vy *= 0.82;

        // Limit maximum velocity
        const maxVelocity = 8;
        const velocity = Math.sqrt(this.vx * this.vx + this.vy * this.vy);

        if (velocity > maxVelocity) {
            this.vx = (this.vx / velocity) * maxVelocity;
            this.vy = (this.vy / velocity) * maxVelocity;
        }

        // Apply velocity
        this.x += this.vx;
        this.y += this.vy;
    }

    render() {
        this.element.style.left = `${this.x}px`;
        this.element.style.top = `${this.y}px`;
    }
}

// Run several small physics steps per frame to make collisions smoother.
function updatePhysics() {
    const physicsSteps = 4;
    for (let i = 0; i < physicsSteps; i++) {
        for (const word of cloudWords) {
            word.update();
        }
    }
}

// Render the current positions
function animate() {
    updatePhysics();
    for (const word of cloudWords) {
        word.render();
    }
    requestAnimationFrame(animate);
}

animate();


// Start game
async function startGame(mode) {
    console.log("Starting game:", mode);

    // Set game mode colour
    themeHue = modeHues[mode];

    // reset gamestate variables
    isPaused = false;
    clearInterval(timerInterval);
    currentMode = mode;

    // If this mode has already been completed, reopen the saved game.
    if (completedGames.has(mode)) {
        loadCompletedGame(mode);
        return;
    }

    gameOver = false;

    // Generate colour scheme
    const backgroundColour = `hsl(${themeHue}, 60%, 92%)`;
    const textColour = `hsl(${themeHue}, 60%, 25%)`;

    document.documentElement.style.setProperty("--background-colour", backgroundColour);
    document.documentElement.style.setProperty("--text-colour", textColour);

    // Generate secret word
    const themes = Object.keys(wordThemes);
    const randomTheme = themes[Math.floor(Math.random() * themes.length)];
    const themeWords = wordThemes[randomTheme];
    secretWord = themeWords[Math.floor(Math.random() * themeWords.length)];
    console.log("Theme:", randomTheme);
    console.log("Secret word:", secretWord);

    // Reset game state
    guesses = new Set();
    cloudWords = [];

    // Create game UI
    app.innerHTML = `
        <div class="game-controls" id="gameControls">
            <button id="menuButton">Menu</button>
            <button id="pauseButton">Pause</button>
            <button id="giveUpButton">Give up</button>
        </div>

        <div class="timer" id="timer">00:00</div>

        <div class="pause-overlay" id="pauseOverlay">
            <div class="pause-box">
                <h2>Game paused</h2>
                <p id="pauseTime">Time: 00:00</p>
                <p id="pauseGuesses">Guesses: 0</p>
                <button id="resumeButton">Resume</button>
            </div>
        </div>

        <main class="game">

            <header class="header">
                <h1>Semantic Word Game</h1>
                <p>Find the secret word.</p>
            </header>

            <section
                class="word-cloud"
                id="wordCloud">
            </section>

            <form
                class="input-area"
                id="guessForm">
                <input
                    type="text"
                    id="wordInput"
                    placeholder="Enter a word..."
                    autocomplete="off"
                    spellcheck="false"
                />
            </form>

            <div
                class="win-message"
                id="winMessage">
                <h2>Congratulations!</h2>
                <p id="winText"></p>
            </div>
        </main>
    `;


    // Get game elements
    wordCloud = document.getElementById("wordCloud");
    guessForm = document.getElementById("guessForm");
    wordInput = document.getElementById("wordInput");
    winMessage = document.getElementById("winMessage");
    winText = document.getElementById("winText");
    timer = document.getElementById("timer");
    const menuButton = document.getElementById("menuButton");
    const pauseButton = document.getElementById("pauseButton");
    const giveUpButton = document.getElementById("giveUpButton");
    const pauseOverlay = document.getElementById("pauseOverlay");
    const pauseTime = document.getElementById("pauseTime");
    const pauseGuesses = document.getElementById("pauseGuesses");
    const resumeButton = document.getElementById("resumeButton");

    menuButton.addEventListener("click", () => {
        showMenu();
    });

    pauseButton.addEventListener("click", () => {
        if (gameOver) {return;}
        isPaused = true;
        clearInterval(timerInterval);
        pausedElapsedTime = Date.now() - startTime;
        pauseTime.textContent = `Time: ${timer.textContent}`;
        const playerGuesses = mode === "hard" ? guesses.size : guesses.size - 1;
        pauseGuesses.textContent = `Guesses: ${playerGuesses}`;
        pauseOverlay.classList.add("visible");
    });

    resumeButton.addEventListener("click", () => {
        isPaused = false;
        pauseOverlay.classList.remove("visible");
        if (gameOver) {return;}
        startTime = Date.now() - pausedElapsedTime;
        timerInterval = setInterval(() => {updateTimer();}, 1000);
    });

    giveUpButton.addEventListener("click", () => {
        if (gameOver) {return;}
        gameOver = true;
        clearInterval(timerInterval);
        const elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
        const minutes = Math.floor(elapsedSeconds / 60);
        const seconds = elapsedSeconds % 60;
        const finalTime = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
        const playerGuesses = currentMode === "hard" ? guesses.size : guesses.size - 1;

        const secretCloudWord = new CloudWord(secretWord, MAX_FONT_SIZE, 1);
        cloudWords.push(secretCloudWord);

        winText.textContent = `The word was "${secretWord}" — you gave up after ${playerGuesses} guesses in ${finalTime}.`;
        winMessage.querySelector("h2").textContent = "Game over";
        winMessage.classList.add("visible");
        wordInput.disabled = true;

        saveCompletedGame(mode);
    });


    // Start timer
    startTime = Date.now();
    timerInterval = setInterval(() => {updateTimer();}, 1000);

    // Starting clue
    if (mode === "easy" || mode === "medium") {
        let clueWord;

        if (mode === "easy") {
            // Easy: give the player the theme itself
            clueWord = randomTheme;
        } else {
            // Medium: give the player a random word from the theme
            const clueWords = themeWords.filter(word => word !== secretWord);
            clueWord = clueWords[Math.floor(Math.random() * clueWords.length)];
        }

        const clueEmbedding = await getEmbedding(clueWord);
        const secretEmbedding = await getEmbedding(secretWord);
        const similarity = cosineSimilarity(clueEmbedding, secretEmbedding);
        const fontSize = similarityToFontSize(similarity);
        const cloudWord = new CloudWord(clueWord, fontSize, similarity);

        cloudWords.push(cloudWord);
        guesses.add(clueWord);
    }

    // Handle guesses
    guessForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            if (isPaused || gameOver) {return;}
            const word = wordInput.value.trim().toLowerCase();

            if (!word) {
                wordInput.value = "";
                return;
            }

            if (!isValidWord(word)) {
                wordInput.value = "";
                return;
            }

            if (guesses.has(word)) {
                wordInput.value = "";
                return;
            }

            const wordEmbedding = await getEmbedding(word);
            const secretEmbedding = await getEmbedding(secretWord);
            const similarity = cosineSimilarity(wordEmbedding, secretEmbedding);
            const fontSize = similarityToFontSize(similarity);
            const cloudWord = new CloudWord(word, fontSize, similarity);

            cloudWords.push(cloudWord);

            if (similarity >= 0.9999) {
                gameOver = true;
                clearInterval(timerInterval);
                completedModes.add(mode);

                const elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
                const minutes = Math.floor(elapsedSeconds / 60);
                const seconds = elapsedSeconds % 60;
                const finalTime = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
                const numGuesses = mode === 'hard' ? guesses.size + 1 : guesses.size;

                winText.textContent = `The word was "${secretWord}" — you got it in ${numGuesses} guesses in ${finalTime}!`;

                winMessage.classList.add("visible");
                wordInput.disabled = true;

                saveCompletedGame(mode);
            }

            wordInput.value = "";
            guesses.add(word);
        }
    );
}


// Save a completed game so it can be reopened during the session.
function saveCompletedGame(mode) {
    completedModes.add(mode);

    completedGames.set(mode, {
        mode: mode,
        secretWord: secretWord,
        guesses: new Set(guesses),
        cloudWords: cloudWords.map(word => ({
            text: word.text,
            fontSize: word.fontSize,
            similarity: word.similarity,
            x: word.x,
            y: word.y
        })),
        finalTime: timer.textContent
    });
}


// Load a completed game from the saved session data.
function loadCompletedGame(mode) {
    const savedGame = completedGames.get(mode);

    themeHue = modeHues[mode];
    secretWord = savedGame.secretWord;
    currentMode = mode;
    gameOver = true;
    isPaused = false;
    guesses = new Set(savedGame.guesses);
    cloudWords = [];

    const backgroundColour = `hsl(${themeHue}, 60%, 92%)`;
    const textColour = `hsl(${themeHue}, 60%, 25%)`;

    document.documentElement.style.setProperty("--background-colour", backgroundColour);
    document.documentElement.style.setProperty("--text-colour", textColour);

    app.innerHTML = `
        <div class="game-controls" id="gameControls">
            <button id="menuButton">Menu</button>
        </div>

        <div class="timer" id="timer">${savedGame.finalTime}</div>

        <main class="game">

            <header class="header">
                <h1>Semantic Word Game</h1>
                <p>Completed game.</p>
            </header>

            <section
                class="word-cloud"
                id="wordCloud">
            </section>

            <form
                class="input-area"
                id="guessForm">
                <input
                    type="text"
                    id="wordInput"
                    placeholder="Game completed"
                    autocomplete="off"
                    spellcheck="false"
                    disabled
                />
            </form>

            <div
                class="win-message visible"
                id="winMessage">
                <h2>Congratulations!</h2>
                <p id="winText"></p>
            </div>
        </main>
    `;

    wordCloud = document.getElementById("wordCloud");
    guessForm = document.getElementById("guessForm");
    wordInput = document.getElementById("wordInput");
    winMessage = document.getElementById("winMessage");
    winText = document.getElementById("winText");
    timer = document.getElementById("timer");

    const menuButton = document.getElementById("menuButton");

    menuButton.addEventListener("click", () => {
        showMenu();
    });

    for (const savedWord of savedGame.cloudWords) {
        const cloudWord = new CloudWord(
            savedWord.text,
            savedWord.fontSize,
            savedWord.similarity,
            savedWord.x,
            savedWord.y
        );

        cloudWords.push(cloudWord);
    }

    const playerGuesses = mode === "hard" ? guesses.size : guesses.size - 1;
    winText.textContent = `The word was "${secretWord}" — you got it in ${playerGuesses} guesses in ${savedGame.finalTime}!`;
}