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

// daily puzzle storage
const DAILY_GAME_KEY = "semanticGravityDailyGames";
const STATS_KEY = "semanticGravityStats";

// word data
const response = await fetch("./words.json");
const wordThemes = await response.json();

// valid words
const validWords = new Set(words.getAll());

function isValidWord(word) {
    return validWords.has(word)
};

const guesses = new Set();

// gamestate
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

// daily puzzle functionality

// get current date
function getToday() {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, "0");
    const day = String(now.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
}

// simple hash to turn date to repeatable number
function hashString(string) {
    let hash = 0;
    for (let i = 0; i < string.length; i++) {
        hash = ((hash << 5) - hash) + string.charCodeAt(i);
        hash |= 0;
    }
    return Math.abs(hash);
}

// generate todays puzzle
function generateDailyPuzzle(mode) {
    const today = getToday();
    const seed = hashString(`${today}-${mode}`);
    const themes = Object.keys(wordThemes);
    const themeIndex = seed % themes.length;
    const theme = themes[themeIndex];
    const themeWords = wordThemes[theme];
    const wordIndex = Math.floor(seed / themes.length) % themeWords.length;
    const secret = themeWords[wordIndex];
    return {
        date: today,
        theme: theme,
        secretWord: secret
    };
}

// get all saved daily games
function getDailyGames() {
    const saved = localStorage.getItem(DAILY_GAME_KEY);
    if (!saved) {
        return {};
    }
    try {
        return JSON.parse(saved);
    } catch {
        return {};
    }
}

// save all daily games
function saveDailyGames(games) {
    localStorage.setItem(DAILY_GAME_KEY, JSON.stringify(games));
}

// get todays saved game for a difficulty
function getSavedDailyGame(mode) {
    const games = getDailyGames();
    const today = getToday();
    if (
        games[mode] &&
        games[mode].date === today
    ) {
        return games[mode];
    }
    return null;
}

// save todays completed game
function saveDailyGame(mode, elapsedTime, completed = true) {
    const games = getDailyGames();
    const savedCloudWords = cloudWords.map(word => {
        return {
            text: word.text,
            fontSize: word.fontSize,
            similarity: word.similarity,
            x: word.x,
            y: word.y
        };
    });

    games[mode] = {
        date: getToday(),
        secretWord: secretWord,
        theme: currentTheme,
        guesses: [...guesses],
        cloudWords: savedCloudWords,
        elapsedTime: elapsedTime,
        completed: completed
    };
    saveDailyGames(games);

    // keep current-session map working too
    completedGames.set(mode, games[mode]);
    completedModes.add(mode);
}

// stats
function getStats() {
    const saved = localStorage.getItem(STATS_KEY);
    if (!saved) {
        return {
            easy: [],
            medium: [],
            hard: []
        };
    }
    try {
        const stats = JSON.parse(saved);
        return {
            easy: stats.easy || [],
            medium: stats.medium || [],
            hard: stats.hard || []
        };
    } catch {
        return {
            easy: [],
            medium: [],
            hard: []
        };
    }
}

function saveStats(stats) {
    localStorage.setItem(STATS_KEY, JSON.stringify(stats));
}

// add completed game to statistics
function recordStats(mode, guessesTaken, elapsedTime) {
    const stats = getStats();
    const today = getToday();

    // don't record same daily puzzle twice
    const alreadyRecorded = stats[mode].some(result => result.date === today);
    if (alreadyRecorded) {
        return;
    }
    stats[mode].push({date: today, guesses: guessesTaken, time: elapsedTime});

    // keep chronological order.
    stats[mode].sort((a, b) => a.date.localeCompare(b.date));
    saveStats(stats);
}

// calculate current solved streak
function getCurrentStreak(mode) {
    const stats = getStats();
    const results = stats[mode];
    if (results.length === 0) {
        return 0;
    }
    const solvedDates = new Set(results.map(result => result.date));
    let streak = 0;
    const date = new Date();

    while (true) {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, "0");
        const day = String(date.getDate()).padStart(2, "0");
        const dateString = `${year}-${month}-${day}`;
        if (!solvedDates.has(dateString)) {
            break;
        }
        streak++;
        date.setDate(date.getDate() - 1);
    }
    return streak;
}

// Calculate the best streak ever
function getBestStreak(mode) {
    const stats = getStats();
    const results = stats[mode];

    if (results.length === 0) {
        return 0;
    }

    let best = 0;
    let current = 0;
    let previousDate = null;

    for (const result of results) {
        const date = new Date(`${result.date}T00:00:00`);
        if (previousDate) {
            const difference = (date - previousDate) / (1000 * 60 * 60 * 24);
            if (difference === 1) {
                current++;
            } else {
                current = 1;
            }
        } else {
            current = 1;
        }
        best = Math.max(best, current);
        previousDate = date;
    }
    return best;
}

function formatTime(milliseconds) {
    const totalSeconds = Math.floor(milliseconds / 1000)
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function getAverage(mode, property) {
    const stats = getStats();
    const results = stats[mode];
    if (results.length === 0) {
        return null;
    }
    const total = results.reduce((sum, result) => sum + result[property], 0);
    return total / results.length;
}

// making stat graphs
function createStatsGraph(mode, property) {
    const stats = getStats();
    const results = stats[mode].slice(-14);

    if (results.length < 2) {
        return `
            <div class="stats-no-graph">
                Complete more daily games to see your progress.
            </div>
        `;
    }

    const width = 600;
    const height = 220;

    const paddingLeft = 45;
    const paddingRight = 20;
    const paddingTop = 20;
    const paddingBottom = 35;

    const values = results.map(result => result[property]);

    const min = Math.min(...values);
    const max = Math.max(...values);

    const range = max - min || 1;

    const points = results.map((result, index) => {
        const x = paddingLeft + (index / (results.length - 1)) * (width - paddingLeft - paddingRight);
        const y = paddingTop + (1 - ((result[property] - min) / range)) * (height - paddingTop - paddingBottom);
        return {x, y, date: result.date, value: result[property]};
    });
    const pointString = points.map(point => `${point.x},${point.y}`).join(" ");
    const circles = points.map(point => `
            <circle
                cx="${point.x}"
                cy="${point.y}"
                r="4"
                class="stats-graph-point">
                <title>
                    ${point.date}: ${
                        property === "time"
                            ? formatTime(point.value)
                            : point.value
                    }
                </title>
            </circle>
        `).join("");

    const labels = points.map((point, index) => {
            if (
                index !== 0 &&
                index !== points.length - 1 &&
                index % 2 !== 0
            ) {
                return "";
            }
            const date = new Date(`${point.date}T00:00:00`);
            return `
                <text
                    x="${point.x}"
                    y="${height - 10}"
                    text-anchor="middle"
                    class="stats-graph-label">
                    ${date.getDate()}/${date.getMonth() + 1}
                </text>
            `;
        }).join("");

    return `
        <svg
            class="stats-graph"
            viewBox="0 0 ${width} ${height}"
            preserveAspectRatio="none">
            <polyline points="${pointString}" class="stats-graph-line" fill="none"
            />
            ${circles}
            ${labels}
        </svg>
    `;
}

// maths/data/library stuff
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
    timer.textContent = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function similarityToFontSize(similarity) {
    if (similarity < MIN_SIMILARITY) {
        return MIN_FONT_SIZE;
    }
    if (similarity > MAX_SIMILARITY) {
        return MAX_FONT_SIZE;
    }
    const x = (similarity - MIN_SIMILARITY) / (MAX_SIMILARITY - MIN_SIMILARITY);
    const curved = 3 * x ** 2 - 2 * x ** 3;
    return (MIN_FONT_SIZE + curved * (MAX_FONT_SIZE - MIN_FONT_SIZE));
};

const extractor = await pipeline(
    "feature-extraction",
    "Xenova/paraphrase-MiniLM-L3-v2",
    {dtype: "q4"}
);

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

// menu ui
function showMenu() {
    clearInterval(timerInterval);
    themeHue = 210;
    const backgroundColour = `hsl(${themeHue}, 60%, 92%)`;
    const textColour = `hsl(${themeHue}, 60%, 25%)`;
    document.documentElement.style.setProperty("--background-colour", backgroundColour);
    document.documentElement.style.setProperty("--text-colour", textColour);

    // check today's completed levels.
    completedModes.clear();
    for (const mode of ["easy", "medium", "hard"]) {
        if (getSavedDailyGame(mode)) {
            completedModes.add(mode);
        }
    }

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
            <button class="tutorial-button" id="tutorialButton">[how to play]</button>
            <button class="tutorial-button" id="statsButton">[statistics]</button>
        </main>
        <div
            class="tutorial-overlay"
            id="tutorialOverlay">
            <div class="tutorial">
                <button class="tutorial-close" id="tutorialClose">x</button>
                <div class="tutorial-image">
                    <img id="tutorialImage" src="./tutorial-1.png" alt="">
                </div>
                <div class="tutorial-content">
                    <h2 id="tutorialTitle"></h2>
                    <p id="tutorialText"></p>
                </div>
                <div class="tutorial-navigation">
                    <button class="tutorial-navigation-button" id="tutorialPrevious">←</button>
                    <span id="tutorialCounter">1 / 6</span>
                    <button class="tutorial-navigation-button" id="tutorialNext">→</button>
                </div>
            </div>
        </div>
    `;

    // level select buttons
    document.querySelectorAll(".mode-button")
        .forEach(button => {

            const mode = button.dataset.mode;

            if (completedModes.has(mode)) {
                button.classList.add("complete");
            }

            button.addEventListener("click", () => {
                startGame(mode);
            });
        });

    // tutorial
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
            text: `This game purely takes into account semantic meaning - although some words may be spelt similarly, if they don't mean the same thing as the secret word, they will be small.`
        },
        {
            title: "One word, many meanings",
            text: `Alternatively, a secret word can link to multiple words that have nothing to do with each other independently. In the example above, the secret word, "pupil", links to both "eye" and "school".`
        },
        {
            title: "Don't go down rabbit holes",
            text: `Don't go too far into rabbit holes when the words are small! In the above example, even though "breakfast" appears large and the given theme was "food", don't go down a rabbit hole of guessing breakfast foods - in this example, the link to breakfast was that, like the secret word "dinner", it is a mealtime.`
        },
        {
            title: "Three levels to master",
            text: `When playing Semantic Gravity, you will have 3 levels to complete. "Easy" will provide you with the theme of the secret word as a starter word. "Medium" will provide you with a word derived from the same theme as the secret word as a starter word. "Hard" will give you no clues, and is by far the most difficult - don't get disheartened if you struggle to find even one non-small word for a while!`
        },
        {
            title: "Follow the cloud",
            text: `I hope this introduction to Semantic Gravity allows you to enjoy the game to the fullest extent, so have fun finding those words in the fewest guesses, or the shortest time, your call! Have fun, and follow the cloud - good luck!`
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

        if (
            tutorialSlide === tutorialSlides.length - 1
        ) {
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
        if (
            tutorialSlide <
            tutorialSlides.length - 1
        ) {
            tutorialSlide++;
            updateTutorial();
        } else {
            tutorialOverlay.classList.remove("visible");
        }});
    updateTutorial();

    // stats button
    const statsButton = document.getElementById("statsButton");
    statsButton.addEventListener("click", () => {showStats();});
}

// stats ui
function showStats() {
    clearInterval(timerInterval);
    themeHue = 210;
    const backgroundColour = `hsl(${themeHue}, 60%, 92%)`;
    const textColour = `hsl(${themeHue}, 60%, 25%)`;
    document.documentElement.style.setProperty("--background-colour", backgroundColour);
    document.documentElement.style.setProperty("--text-colour", textColour);

    app.innerHTML = `
        <main class="home stats-page">
            <header class="header">
                <h1>Statistics</h1>
                <p>Your Semantic Gravity journey.</p>
            </header>

            <div class="stats-modes">
                ${createStatsSection("easy", "Easy")}
                ${createStatsSection("medium", "Medium")}
                ${createStatsSection("hard", "Hard")}
            </div>

            <button class="tutorial-button" id="statsBackButton">[back]</button>
        </main>
    `;

    document.getElementById("statsBackButton").addEventListener("click", () => {showMenu();});
}

// stats panels
function createStatsSection(mode, title) {
    const stats = getStats();
    const results = stats[mode];
    const total = results.length;
    const currentStreak = getCurrentStreak(mode);
    const bestStreak = getBestStreak(mode);
    const bestGuesses = total > 0 ? Math.min(...results.map(result => result.guesses)) : null;
    const bestTime = total > 0 ? Math.min(...results.map(result => result.time)) : null;
    const averageGuesses = getAverage(mode, "guesses");
    const averageTime = getAverage(mode, "time");

    return `
        <section class="stats-section">
            <h2>${title}</h2>

            <div class="stats-grid">
                <div class="stat">
                    <strong>${total}</strong>
                    <span>Solved</span>
                </div>

                <div class="stat">
                    <strong>${currentStreak}</strong>
                    <span>Current streak</span>
                </div>

                <div class="stat">
                    <strong>${bestStreak}</strong>
                    <span>Best streak</span>
                </div>

                <div class="stat">
                    <strong>
                        ${bestGuesses ?? "-"}
                    </strong>
                    <span>Fewest guesses</span>
                </div>

                <div class="stat">
                    <strong>
                        ${bestTime !== null ? formatTime(bestTime) : "-—"}
                    </strong>
                    <span>Fastest time</span>
                </div>

                <div class="stat">
                    <strong>
                        ${averageGuesses !== null ? averageGuesses.toFixed(1) : "-"}
                    </strong>
                    <span>Average guesses</span>
                </div>

                <div class="stat">
                    <strong>
                        ${averageTime !== null ? formatTime(averageTime) : "-"}
                    </strong>
                    <span>Average time</span>
                </div>
            </div>

            <div class="stats-graphs">
                <h3>Guesses over time</h3>
                ${createStatsGraph(mode, "guesses")}

                <h3>Time over time</h3>
                ${createStatsGraph(mode, "time")}
            </div>
        </section>
    `;
}

// plhysics egine
class CloudWord {
    constructor(
        text,
        fontSize,
        similarity,
        savedPosition = null
    ) {
        this.text = text;
        this.fontSize = fontSize;
        this.similarity = similarity;

        // spawn position
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
            const distance = Math.sqrt(Math.random()) *spawnRadius;
            this.x = centreX + Math.cos(angle) * distance;
            this.y = centreY + Math.sin(angle) * distance;
        }

        // velocity
        this.vx = 0;
        this.vy = 0;

        // physical properties
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

        // attraction towards centre
        const halfWidth = this.width / 2;
        const halfHeight = this.height / 2;
        const closestX = Math.max(this.x - halfWidth, Math.min(centreX, this.x + halfWidth));
        const closestY = Math.max(this.y - halfHeight, Math.min(centreY, this.y + halfHeight));
        let centreDX = centreX - closestX;
        let centreDY = centreY - closestY;
        const centreDistance = Math.sqrt(centreDX * centreDX + centreDY * centreDY);

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

        // repulsion from other words
        for (const other of cloudWords) {
            if (other === this) {continue;}
            const dx = this.x - other.x;
            const dy = this.y - other.y;
            const overlapX = (this.width + other.width) / 2 - Math.abs(dx);
            const overlapY = (this.height + other.height) / 2 - Math.abs(dy);
            if (overlapX <= 0 || overlapY <= 0) {continue;}

            // find out which dir needs the least movement
            if (
                overlapX <
                overlapY * 1.3
            ) {
                const direction = dx >= 0 ? 1 : -1;
                const push = overlapX * 0.25;
                forceX += direction * push * (other.mass / this.mass);
            } else {
                const direction = dy >= 0 ? 1 : -1;
                const push = overlapY * 0.25;
                forceY += direction * push * (other.mass / this.mass);
            }

            // Sideways force
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance > 0) {
                const tangentX = -dy / distance;
                const tangentY = dx / distance;
                const slide = Math.min(overlapX, overlapY) * 0.08;
                forceX += tangentX * slide * (other.mass / this.mass);
                forceY += tangentY * slide * (other.mass / this.mass);
            }
        }

        // convert force to acceleration
        this.vx += forceX / this.mass;
        this.vy += forceY / this.mass;

        // slow down
        this.vx *= 0.82;
        this.vy *= 0.82;

        // limit max vel
        const maxVelocity = 8;
        const velocity = Math.sqrt(this.vx * this.vx + this.vy * this.vy);

        if (velocity > maxVelocity) {
            this.vx = (this.vx / velocity) * maxVelocity;
            this.vy = (this.vy / velocity) * maxVelocity;
        }

        // apply vel
        this.x += this.vx;
        this.y += this.vy;
    }

    render() {
        this.element.style.left = `${this.x}px`;
        this.element.style.top = `${this.y}px`;
    }
}

// run a few physics steps per frame
function updatePhysics() {
    const physicsSteps = 4;
    for (
        let i = 0;
        i < physicsSteps;
        i++
    ) {
        for (const word of cloudWords) {
            word.update();
        }
    }
}

// render urrent positions
function animate() {
    updatePhysics();
    for (const word of cloudWords) {
        word.render();
    }
    requestAnimationFrame(animate);
}

animate();

// load already-completed game
function loadCompletedGame(mode, savedGame) {
    console.log("Loading completed game:", mode);
    themeHue = modeHues[mode];
    isPaused = false;
    gameOver = true;
    clearInterval(timerInterval);
    currentMode = mode;

    // restore game data
    secretWord = savedGame.secretWord;
    currentTheme = savedGame.theme;

    guesses.clear();

    for (const guess of savedGame.guesses) {
        guesses.add(guess);
    }

    // Colour scheme
    const backgroundColour = `hsl(${themeHue}, 60%, 92%)`;
    const textColour = `hsl(${themeHue}, 60%, 25%)`;

    document.documentElement.style.setProperty("--background-colour", backgroundColour);
    document.documentElement.style.setProperty("--text-colour", textColour);

    cloudWords = [];

    // menu ui
    app.innerHTML = `
        <div class="game-controls">
            <button id="menuButton">
                Menu
            </button>

        </div>

        <div class="timer" id="timer">
            00:00
        </div>

        <main class="game">
            <header class="header">
                <h1>Semantic Gravity</h1>
                <p>Today's puzzle.</p>
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
                <h2>Level completed!</h2>
                <p id="winText"></p>
            </div>
        </main>
    `;

    // get elements
    wordCloud = document.getElementById("wordCloud");
    guessForm = document.getElementById("guessForm");
    wordInput = document.getElementById("wordInput");
    winMessage = document.getElementById("winMessage");
    winText = document.getElementById("winText");
    timer = document.getElementById("timer");

    const menuButton = document.getElementById("menuButton");
    menuButton.addEventListener("click", () => {showMenu();});

    // restore timer
    timer.textContent = formatTime(savedGame.elapsedTime);

    // restore cloud
    for (
        const savedWord of
        savedGame.cloudWords
    ) {
        const cloudWord = new CloudWord(savedWord.text, savedWord.fontSize, savedWord.similarity, {x: savedWord.x, y: savedWord.y});
        cloudWords.push(cloudWord);
    }

    // sisplay info
    const numGuesses = mode === "hard" ? guesses.size : guesses.size - 1;
    winText.textContent = `The word was "${secretWord}" — you got it in ${numGuesses} guesses in ${timer.textContent}!`;
}

// start game
async function startGame(mode) {
    console.log("Starting game:", mode);

    // check if daily game has already been done
    const savedGame = getSavedDailyGame(mode);

    if (savedGame) {
        completedGames.set(mode, savedGame);
        loadCompletedGame(mode, savedGame);
        return;
    }

    // set game mode colour
    themeHue = modeHues[mode];

    // reset game state
    isPaused = false;
    gameOver = false;

    clearInterval(timerInterval);
    currentMode = mode;

    // colour scheme
    const backgroundColour = `hsl(${themeHue}, 60%, 92%)`;
    const textColour = `hsl(${themeHue}, 60%, 25%)`;

    document.documentElement.style.setProperty("--background-colour", backgroundColour);
    document.documentElement.style.setProperty("--text-colour", textColour);

    // generate daily words
    const dailyPuzzle = generateDailyPuzzle(mode);
    currentTheme = dailyPuzzle.theme;
    secretWord = dailyPuzzle.secretWord;
    console.log("Today's date:", dailyPuzzle.date);

    // Reset game state
    guesses.clear();
    cloudWords = [];

    // game ui
    app.innerHTML = `
        <div
            class="game-controls"
            id="gameControls">
            <button id="menuButton">Menu</button>
            <button id="pauseButton">Pause</button>
            <button id="giveUpButton">Give up</button>
        </div>

        <div
            class="timer"
            id="timer">
            00:00
        </div>

        <div
            class="pause-overlay"
            id="pauseOverlay">
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
                <p>Today's puzzle.</p>
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

    // Get elements
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

    // return to menu
    menuButton.addEventListener(
        "click",
        () => {
            showMenu();
        }
    );

    // pause
    pauseButton.addEventListener(
        "click",
        () => {
            isPaused = true;
            clearInterval(timerInterval);
            pausedElapsedTime = Date.now() - startTime;
            pauseTime.textContent = `Time: ${timer.textContent}`;
            const playerGuesses = mode === "hard" ? guesses.size : guesses.size - 1;
            pauseGuesses.textContent = `Guesses: ${playerGuesses}`;
            pauseOverlay.classList.add("visible");
        }
    );

    // resume
    resumeButton.addEventListener(
        "click",
        () => {
            isPaused = false;
            pauseOverlay.classList.remove("visible");
            if (gameOver) {return;}
            startTime = Date.now() - pausedElapsedTime;
            timerInterval = setInterval(() => {updateTimer();}, 1000);
        }
    );

    // give up
    giveUpButton.addEventListener(
        "click",
        () => {
            if (gameOver) {return;}
            gameOver = true;
            clearInterval(timerInterval);

            const elapsedTime = Date.now() - startTime;
            const finalTime = formatTime(elapsedTime);
            const playerGuesses = currentMode === "hard" ? guesses.size : guesses.size - 1;
            const secretCloudWord = new CloudWord(secretWord, MAX_FONT_SIZE, 1);

            cloudWords.push(secretCloudWord);
            winText.textContent = `The word was "${secretWord}" — you gave up after ${playerGuesses} guesses in ${finalTime}.`;

            winMessage.querySelector("h2").textContent = "Game over";
            winMessage.classList.add("visible");

            wordInput.disabled = true;
            saveDailyGame(currentMode, elapsedTime, false);
        }
    );

    // timer
    startTime = Date.now();
    timerInterval = setInterval( () => {updateTimer();}, 1000);

    // starting clue
    if (
        mode === "easy" ||
        mode === "medium"
    ) {
        let clueWord;

        if (mode === "easy") {
            clueWord = currentTheme;
        } else {
            const themeWords = wordThemes[currentTheme];
            const clueWords = themeWords.filter(word => word !== secretWord);
            clueWord = clueWords[hashString(`${getToday()}-${mode}-clue`) % clueWords.length];
        }

        const clueEmbedding = await getEmbedding(clueWord);
        const secretEmbedding = await getEmbedding(secretWord);
        const similarity = cosineSimilarity(clueEmbedding, secretEmbedding);
        const fontSize = similarityToFontSize(similarity);
        const cloudWord = new CloudWord(clueWord, fontSize, similarity);

        cloudWords.push(cloudWord);
        guesses.add(clueWord);
    }

    // handle guesss
    guessForm.addEventListener(
        "submit",
        async event => {
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

            // win condition
            if (similarity >= 0.9999) {
                gameOver = true;
                clearInterval(timerInterval);

                const elapsedTime = Date.now() - startTime;
                const numGuesses = mode === "hard" ? guesses.size + 1 : guesses.size;

                if (savedGame.completed) {
                    winMessage.querySelector("h2").textContent = "Level completed!";
                    winText.textContent = `The word was "${secretWord}" — you got it in ${numGuesses} guesses in ${timer.textContent}!`;
                } else {
                    winMessage.querySelector("h2").textContent = "Game over";
                    winText.textContent = `The word was "${secretWord}" — you gave up after ${numGuesses} guesses in ${timer.textContent}.`;
                }

                winMessage.classList.add("visible");
                wordInput.disabled = true;

                // Save today's completed game
                saveDailyGame(mode, elapsedTime);

                // Record statistics
                recordStats(mode, numGuesses, elapsedTime);
            }

            wordInput.value = "";
            guesses.add(word);
        }
    );
}