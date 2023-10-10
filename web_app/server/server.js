import express from 'express';
import { exec } from 'child_process';
import cors from 'cors';

const app = express();
const PORT = 5002;

// Use the cors middleware
app.use(cors());

app.use(express.json());

app.post('/execute', (req, res) => {
    const prompt = req.body.prompt;

    exec(`./run model.bin -i '${prompt}'`, (error, stdout, stderr) => {
        if (error) {
            console.error(`exec error: ${error}`);
            return res.status(500).send(`Error: ${error}`);
        }
        return res.send({ output: stdout });
    });
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});




