console.log("Semantic word game!!!!!") // very important line

import "./style.css";
import { pipeline } from "@huggingface/transformers";
import { words } from 'popular-english-words';

// ooh magic numbers!
const MIN_SIMILARITY = 0.15;
const MAX_SIMILARITY = 0.75;

const MIN_FONT_SIZE = 16;
const MAX_FONT_SIZE = 64;

const WORD_MOVEMENT_SPEED = 0.08; // proportion of remaining dist to target moved each frame (0-1)
const WORD_PADDING_FACTOR = 1.25;

const response = await fetch("./nouns.csv"); const text = await response.text(); const nouns = text.split(/\r?\n/).map(noun => noun.trim()).filter(noun => noun.length > 0);
const secretWord = nouns[Math.floor(Math.random() * nouns.length)];
console.log("Secret word: ", secretWord);

const validWords = new Set(words.getAll());
function isValidWord(word) {return validWords.has(word)};

const guesses = new Set();

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

    return dotProduct / (magnitudeA * magnitudeB);
};

function similarityToFontSize(similarity) {
    const minSize = 16;
    const maxSize = 64;

    if (similarity < MIN_SIMILARITY) {
        return MIN_FONT_SIZE;
    }

    if (similarity > MAX_SIMILARITY) {
        return MAX_FONT_SIZE;
    }

    const x = (similarity - MIN_SIMILARITY) / (MAX_SIMILARITY - MIN_SIMILARITY);

    const curved = 3 * x ** 2 - 2 * x ** 3;

    return minSize + curved * (maxSize - minSize);
};

const extractor = await pipeline(
    "feature-extraction",
    "Xenova/paraphrase-MiniLM-L3-v2",
    {
        dtype: "q4"
    }
);

const embeddingCache = new Map();

async function getEmbedding(word) {
    if (embeddingCache.has(word)) {
        return embeddingCache.get(word);
    }

    const output = await extractor(word, {
        pooling: "mean",
        normalize: true
    });

    embeddingCache.set(word, output);

    return output;
};

// html stuff

const app = document.querySelector("#app");

app.innerHTML = `
    <main class="game">
        <header class="header">
            <h1>Semantic Word</h1>
            <p>Find today's hidden word.</p>
        </header>

        <section class="word-cloud" id="wordCloud"></section>

        <form class="input-area" id="guessForm">
            <input
                type="text"
                id="wordInput"
                placeholder="Enter a word..."
                autocomplete="off"
                spellcheck="false"
            />
        </form>
    </main>
`;

const wordCloud = document.getElementById("wordCloud");
const cloudWords = [];

// physics stuff

class CloudWord {
    constructor(text, fontSize, similarity) {
        this.text = text;
        this.fontSize = fontSize;
        this.similarity = similarity;

        // Position

        this.x = wordCloud.clientWidth / 2;
        this.y = wordCloud.clientHeight / 2;

        // Velocity

        this.vx = 0;
        this.vy = 0;

        // Physical properties

        this.width = 0;
        this.height = 0;

        this.mass = Math.max(1, fontSize / 16);

        this.element = document.createElement("span");
        this.element.textContent = text;
        this.element.classList.add("cloud-word");
        this.element.style.fontSize = `${fontSize}px`;

        if (similarity >= 0.9999) {
            this.element.classList.add("correct");
        }

        wordCloud.appendChild(this.element);

        this.width =
            this.element.offsetWidth * WORD_PADDING_FACTOR;

        this.height =
            this.element.offsetHeight * WORD_PADDING_FACTOR;
    }

    update() {
        const centreX = wordCloud.clientWidth / 2;
        const centreY = wordCloud.clientHeight / 2;

        let forceX = 0;
        let forceY = 0;


        // Attraction towards centre

        const centreDX = centreX - this.x;
        const centreDY = centreY - this.y;

        const centreDistance = Math.sqrt(
            centreDX * centreDX +
            centreDY * centreDY
        );

        if (centreDistance > 0) {

            const attraction =
                0.0005 +
                this.similarity * 0.004;

            forceX +=
                (centreDX / centreDistance) *
                centreDistance *
                attraction;

            forceY +=
                (centreDY / centreDistance) *
                centreDistance *
                attraction;
        }


        // Repulsion from other words

        for (const other of cloudWords) {
            if (other === this) {
                continue;
            }

            const dx = this.x - other.x;
            const dy = this.y - other.y;

            const overlapX =
                (this.width + other.width) / 2 -
                Math.abs(dx);

            const overlapY =
                (this.height + other.height) / 2 -
                Math.abs(dy);

            if (overlapX <= 0 || overlapY <= 0) {
                continue;
            }


            // Which direction requires the least
            // movement to resolve the collision?

            if (overlapX < overlapY * 1.5) {

                const direction =
                    dx >= 0 ? 1 : -1;

                const push =
                    overlapX * 0.25;

                forceX +=
                    direction *
                    push *
                    (other.mass / this.mass);

            } else {

                const direction =
                    dy >= 0 ? 1 : -1;

                const push =
                    overlapY * 0.25;

                forceY +=
                    direction *
                    push *
                    (other.mass / this.mass);
            }


            // Small sideways force allows words
            // to slide around each other rather
            // than forming vertical stacks.

            const distance =
                Math.sqrt(dx * dx + dy * dy);

            if (distance > 0) {

                const tangentX = -dy / distance;
                const tangentY = dx / distance;

                const slide =
                    Math.min(overlapX, overlapY) *
                    0.15;

                forceX +=
                    tangentX *
                    slide *
                    (other.mass / this.mass);

                forceY +=
                    tangentY *
                    slide *
                    (other.mass / this.mass);
            }
        }


        // Convert force into acceleration

        this.vx += forceX / this.mass;
        this.vy += forceY / this.mass;


        // Damping

        this.vx *= 0.75;
        this.vy *= 0.75;


        // Limit maximum velocity

        const maxVelocity = 8;

        const velocity =
            Math.sqrt(
                this.vx * this.vx +
                this.vy * this.vy
            );

        if (velocity > maxVelocity) {

            this.vx =
                (this.vx / velocity) *
                maxVelocity;

            this.vy =
                (this.vy / velocity) *
                maxVelocity;
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


// Run several small physics steps per frame
// to make collisions smoother.

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

// display results

const guessForm = document.getElementById("guessForm");
const wordInput = document.getElementById("wordInput");

guessForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const word = wordInput.value.trim().toLowerCase();

    if (!word) {
        wordInput.value = "";
        return;
    }

    if (!isValidWord(word)) {
        console.log(word, "isn't a word");
        wordInput.value = "";
        return;
    }

    if (guesses.has(word)) {
        console.log(word, "has already been guessed");
        wordInput.value = "";
        return;
    }

    const wordEmbedding = await getEmbedding(word);
    const secretEmbedding = await getEmbedding(secretWord);

    const similarity = cosineSimilarity(
        wordEmbedding,
        secretEmbedding
    );

    const fontSize = similarityToFontSize(similarity);

    const cloudWord = new CloudWord(word, fontSize, similarity);

    cloudWords.push(cloudWord);

    wordInput.value = "";
    guesses.add(word);
});