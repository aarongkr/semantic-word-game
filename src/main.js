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
const guesses = new Set();

// Game state
let themeHue = 240;

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
let currentTheme;
let currentMode;

// Store completed games for the current session
const completedGames = new Map();

// important maths/data/library stuff
function cosineSimilarity(a, b) {
    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;
    for (let i = 0; i < a.data.length; i++) {
        dotProduct += a.data[i] * b.data[i];
        magnitudeA += a.data[i] * a.data[i];
        magnitudeB += b.data[i] * b.data[i];
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
    
    themeHue = 210;

    const backgroundColour = `hsl(${themeHue}, 60%, 92%)`;
    const textColour = `hsl(${themeHue}, 60%, 25%)`;

    document.documentElement.style.setProperty("--background-colour", backgroundColour);
    document.documentElement.style.setProperty("--text-colour", textColour);
    
    app.innerHTML = `
        <main class="home">

            <header class="header">
                <h1>Semantic Gravity</h1>
                <p>Follow the cloud.</p>
            </header>

            <div class="mode-selection">
                <button class="mode-button easy" data-mode="easy">[easy]</button>
                <button class="mode-button medium" data-mode="medium">[medium]</button>
                <button class="mode-button hard" data-mode="hard">[hard]</button>
            </div>

            <button class="tutorial-button" id="tutorialButton">
                [how to play]
            </button>

        </main>

        <div class="tutorial-overlay" id="tutorialOverlay">

            <div class="tutorial">

                <button class="tutorial-close" id="tutorialClose">
                    ×
                </button>

                <div class="tutorial-image">
                    <img id="tutorialImage" src="./tutorial-1.png" alt="">
                </div>

                <div class="tutorial-content">
                    <h2 id="tutorialTitle"></h2>
                    <p id="tutorialText"></p>
                </div>

                <div class="tutorial-navigation">

                    <button
                        class="tutorial-navigation-button"
                        id="tutorialPrevious">
                        ←
                    </button>

                    <span id="tutorialCounter">1 / 6</span>

                    <button
                        class="tutorial-navigation-button"
                        id="tutorialNext">
                        →
                    </button>

                </div>

            </div>

        </div>
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


    // Tutorial
    const tutorialButton = document.getElementById("tutorialButton");
    const tutorialOverlay = document.getElementById("tutorialOverlay");
    const tutorialClose = document.getElementById("tutorialClose");
    const tutorialPrevious = document.getElementById("tutorialPrevious");
    const tutorialNext = document.getElementById("tutorialNext");
    const tutorialImage = document.getElementById("tutorialImage");
    const tutorialTitle = document.getElementById("tutorialTitle");
    const tutorialText = document.getElementById("tutorialText");
    const tutorialCounter = document.getElementById("tutorialCounter");

    const tutorialSlides = [
        {
            title: "Welcome to Semantic Gravity",
            text: `You must guess the secret word based on your word cloud, which grows with each guess. Guesses that are more semantically similar to the secret word will appear larger, and those further away will be smaller.`
        },
        {
            title: "It's all about meaning",
            text: `This game purely takes into account semantic meaning — although some words may be spelt similarly, if they don't mean the same thing as the secret word, they will be small.`
        },
        {
            title: "One word, many meanings",
            text: `Alternatively, a secret word can link to multiple words that have nothing to do with each other independently. In the example above, the secret word, "pupil", links to both "eye" and "school".`
        },
        {
            title: "Don't go down rabbit holes",
            text: `Don't go too far into rabbit holes when the words are small! In the above example, even though "breakfast" appears large and the given theme was "food", don't go down a rabbit hole of guessing breakfast foods — in this example, the link to breakfast was that, like the secret word "dinner", it is a mealtime.`
        },
        {
            title: "Three levels to master",
            text: `When playing Semantic Gravity, you will have 3 levels to complete. "Easy" will provide you with the theme of the secret word as a starter word. "Medium" will provide you with a word derived from the same theme as the secret word as a starter word. "Hard" will give you no clues, and is by far the most difficult — don't get disheartened if you struggle to find even one non-small word for a while!`
        },
        {
            title: "Follow the cloud",
            text: `I hope this introduction to Semantic Gravity allows you to enjoy the game to the fullest extent, so have fun finding those words in the fewest guesses, or the shortest time, your call! Have fun, and follow the cloud — good luck!`
        }
    ];

    let tutorialSlide = 0;

    function updateTutorial() {
        const slide = tutorialSlides[tutorialSlide];

        tutorialImage.src = `./tutorial-${tutorialSlide + 1}.png`;
        tutorialTitle.textContent = slide.title;
        tutorialText.textContent = slide.text;
        tutorialCounter.textContent = `${tutorialSlide + 1} / ${tutorialSlides.length}`;

        tutorialPrevious.disabled = tutorialSlide === 0;

        if (tutorialSlide === tutorialSlides.length - 1) {
            tutorialNext.textContent = "Done";
        } else {
            tutorialNext.textContent = "→";
        }
    }

    tutorialButton.addEventListener("click", () => {
        tutorialSlide = 0;
        updateTutorial();
        tutorialOverlay.classList.add("visible");
    });

    tutorialClose.addEventListener("click", () => {
        tutorialOverlay.classList.remove("visible");
    });

    tutorialOverlay.addEventListener("click", event => {
        if (event.target === tutorialOverlay) {
            tutorialOverlay.classList.remove("visible");
        }
    });

    tutorialPrevious.addEventListener("click", () => {
        if (tutorialSlide > 0) {
            tutorialSlide--;
            updateTutorial();
        }
    });

    tutorialNext.addEventListener("click", () => {
        if (tutorialSlide < tutorialSlides.length - 1) {
            tutorialSlide++;
            updateTutorial();
        } else {
            tutorialOverlay.classList.remove("visible");
        }
    });

    updateTutorial();
}

// physics stuff
class CloudWord {
    constructor(text, fontSize, similarity, savedPosition = null, frozen = false) {
        this.text = text;
        this.fontSize = fontSize;
        this.similarity = similarity;
        this.frozen = frozen;

        // Position
        const centreX = wordCloud.clientWidth / 2;
        const centreY = wordCloud.clientHeight / 2;

        if (savedPosition) {
            this.x = savedPosition.x;
            this.y = savedPosition.y;
        } else {
            const maximumRadius = 250;
            const similarityRatio = Math.max(0, Math.min(1, similarity / MAX_SIMILARITY));
            const spawnRadius = (maximumRadius * (1 - similarityRatio)) + 20;
            const angle = Math.random() < 0.5 ? (Math.random() - 0.5) * Math.PI / 1.2 : Math.PI + (Math.random() - 0.5) * Math.PI / 1.2;
            const distance = Math.sqrt(Math.random()) * spawnRadius;
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
        if (this.frozen) {return;}

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


// Save a completed game for the current session
function saveCompletedGame(mode, elapsedTime) {
    const savedCloudWords = cloudWords.map(word => {
        return {
            text: word.text,
            fontSize: word.fontSize,
            similarity: word.similarity,
            x: word.x,
            y: word.y
        };
    });

    completedGames.set(mode, {
        secretWord: secretWord,
        theme: currentTheme,
        guesses: [...guesses],
        cloudWords: savedCloudWords,
        elapsedTime: elapsedTime
    });

    completedModes.add(mode);
}


// Load a previously completed game
function loadCompletedGame(mode, savedGame) {
    console.log("Loading completed game:", mode);

    // Set game mode colour
    if (mode) {
        themeHue = modeHues[mode];
    } else {
        themeHue = 240;
    }
    

    // reset gamestate variables
    isPaused = false;
    gameOver = true;
    clearInterval(timerInterval);
    currentMode = mode;

    // Restore game data
    secretWord = savedGame.secretWord;
    currentTheme = savedGame.theme;

    guesses.clear();
    for (const guess of savedGame.guesses) {
        guesses.add(guess);
    }

    // Generate colour scheme
    const backgroundColour = `hsl(${themeHue}, 60%, 92%)`;
    const textColour = `hsl(${themeHue}, 60%, 25%)`;

    document.documentElement.style.setProperty("--background-colour", backgroundColour);
    document.documentElement.style.setProperty("--text-colour", textColour);

    // Reset word cloud
    cloudWords = [];

    // Create game UI
    app.innerHTML = `
        <div class="game-controls" id="gameControls">
            <button id="menuButton">Menu</button>
        </div>

        <div class="timer" id="timer">00:00</div>

        <main class="game">

            <header class="header">
                <h1>Semantic Gravity</h1>
                <p>Follow the cloud.</p>
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
                    disabled
                />
            </form>

            <div
                class="win-message visible"
                id="winMessage">
                <h2>Level completed!</h2>
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

    menuButton.addEventListener("click", () => {
        showMenu();
    });

    // Restore timer
    const elapsedSeconds = Math.floor(savedGame.elapsedTime / 1000);
    const minutes = Math.floor(elapsedSeconds / 60);
    const seconds = elapsedSeconds % 60;

    timer.textContent =
        `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;

    // Restore word cloud
    for (const savedWord of savedGame.cloudWords) {
        const cloudWord = new CloudWord(
            savedWord.text,
            savedWord.fontSize,
            savedWord.similarity,
            {
                x: savedWord.x,
                y: savedWord.y
            }
        );

        cloudWords.push(cloudWord);
    }

    // Display completion information
    const numGuesses = mode === "hard"
        ? guesses.size
        : guesses.size - 1;

    winText.textContent =
        `The word was "${secretWord}" — you got it in ${numGuesses} guesses in ${timer.textContent}!`;
}


// Start game
async function startGame(mode) {
    console.log("Starting game:", mode);

    // If this level has already been completed, load the saved game
    if (completedGames.has(mode)) {
        loadCompletedGame(mode, completedGames.get(mode));
        return;
    }

    // Set game mode colour
        if (mode) {
            themeHue = modeHues[mode];
        } else {
            themeHue = 240;
        }

    // reset gamestate variables
    isPaused = false;
    gameOver = false;
    clearInterval(timerInterval);
    currentMode = mode;

    // Generate colour scheme
    const backgroundColour = `hsl(${themeHue}, 60%, 92%)`;
    const textColour = `hsl(${themeHue}, 60%, 25%)`;

    document.documentElement.style.setProperty("--background-colour", backgroundColour);
    document.documentElement.style.setProperty("--text-colour", textColour);

    // Generate secret word
    const themes = Object.keys(wordThemes);
    let randomTheme = themes[Math.floor(Math.random() * themes.length)];
    const themeWords = wordThemes[randomTheme];
    secretWord = themeWords[Math.floor(Math.random() * themeWords.length)];
    randomTheme = "food";
    secretWord = "dinner";
    console.log("Theme:", randomTheme);
    console.log("Secret word:", secretWord);

    // Store current theme for persistence
    currentTheme = randomTheme;

    // Reset game state
    guesses.clear();
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
                <h1>Semantic Gravity</h1>
                <p>Follow the cloud.</p>
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

        const playerGuesses = currentMode === "hard"
            ? guesses.size
            : guesses.size - 1;

        const secretCloudWord = new CloudWord(secretWord, MAX_FONT_SIZE, 1);
        cloudWords.push(secretCloudWord);

        winText.textContent =
            `The word was "${secretWord}" — you gave up after ${playerGuesses} guesses in ${finalTime}.`;

        winMessage.querySelector("h2").textContent = "Game over";
        winMessage.classList.add("visible");
        wordInput.disabled = true;
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

                const elapsedTime = Date.now() - startTime;

                const elapsedSeconds = Math.floor(elapsedTime / 1000);
                const minutes = Math.floor(elapsedSeconds / 60);
                const seconds = elapsedSeconds % 60;
                const finalTime = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;

                const numGuesses = mode === 'hard'
                    ? guesses.size + 1
                    : guesses.size;

                winText.textContent =
                    `The word was "${secretWord}" — you got it in ${numGuesses} guesses in ${finalTime}!`;

                winMessage.classList.add("visible");
                wordInput.disabled = true;

                // Save the completed game
                saveCompletedGame(mode, elapsedTime);
            }

            wordInput.value = "";
            guesses.add(word);
        }
    );
}