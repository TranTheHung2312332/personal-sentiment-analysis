import fs from "fs";
import Papa from "papaparse";
import { writeToPath } from "fast-csv";
import { init } from "@heyputer/puter.js/src/init.cjs";
import cliProgress from "cli-progress";

import dotenv from "dotenv"
dotenv.config();

const puter = init(process.env.AUTH_TOKEN);
// puter.ai.chat("Hello", {model: "gpt-4o"}).then(res => console.log(res.message))

const getPrompt = (text) => `
Read the following text and determine whether its sentiment is positive (0), neutral (1), or negative (2):
${text}

Format: Return only a single number: 0, 1, or 2. Do not provide any additional explanation.
 `;

const getLabel = async (text) => {
    const label = await puter.ai.chat(getPrompt(`${text}`), { model: "gpt-4o" });
    
    return Number(label.message.content);
};

async function updateCSV(filePath) {
    try {
        const file = fs.readFileSync(filePath, "utf8");
        const parsed = Papa.parse(file, { header: true, skipEmptyLines: true });
        const data = parsed.data;

        const bar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
        bar.start(data.length, 0);

        for (let i = 0; i < data.length; i++) {
            const row = data[i];

            if (row.is_uncertain === "True") {
                try {
                    const newSentiment = await getLabel(row.text);
                    row.sentiment = Number(newSentiment);
                    row.is_uncertain = "False";
                } catch (err) {
                    console.error(`Error processing row ${i}:`, err);
                    row.sentiment = -1;
                    row.is_uncertain = "True";
                }
            }

            bar.update(i + 1);
        }

        bar.stop();

        const finalData = data.map(({ text, sentiment, is_uncertain }) => ({ text, sentiment, is_uncertain }));

        const ws = fs.createWriteStream("../data/labeled/total.csv");
        writeToPath(ws.path, finalData, { headers: true })
            .on("finish", () => console.log("Complete"));
    } catch (err) {
        console.error("Error reading CSV:", err);
    }
}

updateCSV("../data/log/soft_labeled.csv");
