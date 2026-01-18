import fs from "fs";
import Papa from "papaparse";
import { writeToPath } from "fast-csv";
import { init } from "@heyputer/puter.js/src/init.cjs";
import cliProgress from "cli-progress";

import dotenv from "dotenv"
dotenv.config();

const puter = init(process.env.AUTH_TOKEN);
// puter.ai.chat("Hello", {model: "gpt-4o"}).then(res => console.log(res.message))

const getPrompt = (emoji) => `
You are an emoji classification engine.

Task:
Given the following emoji, return exactly ONE standardized emoji class name.

Emoji:
${emoji}

Output format:
[EMO_<NAME>]

Rules:
- Do NOT use the official Unicode emoji name.
- Do NOT use POSITIVE / NEGATIVE / NEUTRAL or similar sentiment-only labels.
- Focus on the PRIMARY meaning or function of the emoji.
- If the emoji expresses emotion, name the emotion (e.g. HAHA, ANGER, SAD).
- If the emoji does NOT express emotion, classify by meaning:
  action, status, symbol, object, or abstract concept.
- Use concise, meaningful words (1–2 words).
- Uppercase letters and underscore only (A–Z, _).
- Return ONLY the class in brackets. No explanation, no extra text.

Examples:
😂 -> [EMO_HAHA]
😡 -> [EMO_ANGER]
🔕 -> [EMO_MUTE]
🎲 -> [EMO_CHANCE]
♍ -> [EMO_ZODIAC_VIRGO]
⚠️ -> [EMO_WARNING]
`;

const getDesc = async (emoji) => {
    const response = await puter.ai.chat(getPrompt(emoji), { model: "gpt-4o" });
    
    return response.message.content;
};

async function updateCSV(filePath) {
    try {
        const file = fs.readFileSync(filePath, "utf8");
        const parsed = Papa.parse(file, { header: true, skipEmptyLines: true });
        const data = parsed.data;

        const bar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
        bar.start(data.length, 0);

        let errorCount = 0;

        for (let i = 0; i < data.length; i++) {
            const row = data[i];

            if(row['Unicode name'] === '[object Object]'){
                try {                    
                    const desc = await getDesc(row.Emoji);
                    row['Unicode name'] = desc;
                } catch (err) {
                    console.error(`Error processing row ${i}:`, err);
                    row['Unicode name'] = '[EMO]';
                    errorCount ++;
                }
            }

            bar.update(i + 1);
        }

        bar.stop();

        console.log(`error count: ${errorCount}`);

        const ws = fs.createWriteStream(filePath);
        writeToPath(ws.path, data, { headers: true })
            .on("finish", () => console.log("Complete"));
    } catch (err) {
        console.error("Error reading CSV:", err);
    }
}

updateCSV("../data/mapping/emoji.csv");
