console.log("Semantic word game!!!!!") // very important line

import "./style.css";
import { pipeline } from "@huggingface/transformers";

// important maths/data/library stuff

function cosineSimilarity(a, b) {
    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;

    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];

        magnitudeA += a[i] * a[i];
        magnitudeB += b[i] * b[i];
    }

    magnitudeA = Math.sqrt(magnitudeA);
    magnitudeB = Math.sqrt(magnitudeB);

    return dotProduct / (magnitudeA * magnitudeB);
}

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
}

// init test data

const wordPairs = [
    ["apple", "banana"],
    ["apple", "orange"],
    ["apple", "fruit"],
    ["apple", "tree"],
    ["apple", "house"],
    ["apple", "computer"],
    ["dog", "cat"],
    ["dog", "wolf"],
    ["dog", "banana"],
    ["hot", "cold"],
    ["hot", "warm"],
    ["volcano", "lava"],
    ["volcano", "mountain"],
    ["volcano", "rock"],
    ["volcano", "computer"]
];

// html stuff

const app = document.querySelector("#app");

app.innerHTML = `
  <h1>Semantic Word Test</h1>

  <div id="results"></div>
`;


// display results

for (const [word1, word2] of wordPairs) {
    const embedding1 = await getEmbedding(word1);
    const embedding2 = await getEmbedding(word2);

    const similarity = cosineSimilarity(
        embedding1.data,
        embedding2.data
    );

    const result = document.createElement("div");

    result.textContent =
        `${word1} → ${word2}: ${similarity.toFixed(4)}`;

    results.appendChild(result);
}
