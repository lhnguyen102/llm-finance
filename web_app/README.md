# LLM-FRONTEND

## Installation

### Prerequisites
- [Node.js](https://nodejs.org/)

### Install Dependencies
Navigate to the `client` and `server` directories and run `npm install` in each to install the necessary dependencies. 
Use the command `cd client` to switch to the client directory, and `cd ../server` to switch to the server directory.

## Runner
1. Copy trained model (`model.bin`) and binary file (`run.o`) to `web_app/server` folder
2. Open two terminals to run the client and server separately. 

In the first terminal for the client, run:
```shell
cd client
npm run dev
```

In the second terminal for the server, run:

```shell
cd server
npm run server

```
Open your web browser and enter the following URL http://localhost:5173

## Acknowledgement
The code is derived from [the repository](https://github.com/adrianhajdin/project_openai_codex) by Adrian Hajdin - JS Mastery
