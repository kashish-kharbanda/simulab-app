# SimuLab - AI-Powered Drug Discovery Platform

SimuLab is an advanced drug discovery simulation platform that leverages multi-agent AI orchestration to accelerate lead molecule discovery and de-risk R&D.

## 🚀 Features

- **Multi-Agent Orchestration**: Orchestrator, Simulator, and Judge agents work together to evaluate molecular candidates
- **LLM-Powered Analysis**: GPT-4o integration for intelligent molecular evaluation
- **Google Sheets Integration**: Database-backed validation for high-confidence results
- **Interactive Decision Criteria**: Adjustable potency, safety, and cost thresholds
- **Real-time Progress Tracking**: Visual workflow showing agent activity
- **Pareto Analysis**: Multi-objective optimization visualization
- **Edit & Regenerate**: Modify results with natural language commands

## 🛠️ Tech Stack

- **Frontend**: Next.js 15, React 18, TypeScript
- **Styling**: Tailwind CSS, Framer Motion
- **AI**: OpenAI GPT-4o
- **Data**: Google Sheets API integration

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/simulab-app.git
cd simulab-app

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env.local
# Add your OPENAI_API_KEY to .env.local

# Run development server
npm run dev
```

## 🔧 Environment Variables

Create a `.env.local` file with:

```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
```

## 🌐 Deployment

This app is configured for easy deployment on Vercel:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/YOUR_USERNAME/simulab-app)

## 📖 Usage

1. **Define Objective**: Enter your target protein and research goal
2. **Set Constraints**: Specify any molecular constraints (optional)
3. **Review Design**: Examine proposed scenarios and adjust decision criteria
4. **Generate Report**: Run parallel LLM evaluations on all scenarios
5. **Analyze Results**: View Pareto chart, calculation breakdown, and winner selection
6. **Edit & Refine**: Use natural language to modify results if needed

## 🏗️ Architecture

```
┌─────────────────┐
│   Orchestrator  │ ← Manages workflow, pulls scenarios from database
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Simulator    │ ← Runs N parallel evaluations (Potency, Safety, Cost)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      Judge      │ ← Multi-objective analysis, selects winner
└─────────────────┘
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.
