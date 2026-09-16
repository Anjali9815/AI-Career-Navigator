# Frontend structure

```
frontend/
├── index.html
├── package.json
├── vite.config.js
└── src/
    ├── main.jsx              entry point, mounts App and imports the stylesheet
    ├── App.jsx               page state and layout only, no markup detail
    ├── api.js                every backend call, one place to change the URL
    ├── constants.js          example queries
    ├── utils.js              small pure helpers
    ├── styles/
    │   └── index.css         design tokens and all component classes
    └── components/
        ├── Header.jsx        title and subtitle
        ├── SearchBox.jsx     textarea, hint, submit button
        ├── Examples.jsx      clickable starter queries
        ├── Notice.jsx        warn / error banner
        ├── ResultView.jsx    decides what to render for a response
        ├── DirectionPanel.jsx
        ├── MatchList.jsx     header plus the list of cards
        ├── MatchCard.jsx     one person
        ├── Timeline.jsx      their career steps
        ├── SkillTags.jsx     skill pills
        └── NextSteps.jsx     numbered actions
```

## Conventions

**App.jsx holds state, components hold markup.** Every component takes props
and renders. None of them fetch, and none of them own state. That keeps the
data flow in one readable place.

**All styling lives in `styles/index.css`.** Colours, spacing and radii are CSS
custom properties on `:root`, so changing the accent colour is one line rather
than a search across files.

**All network calls go through `api.js`.** Pointing at a deployed backend means
changing one constant, or setting `VITE_API_URL` in a `.env` file.

## Install

Replace the contents of `frontend/src/` with this folder, then set the body tag
in `index.html` to:

```html
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.jsx"></script>
</body>
```

The old inline `style="background:#0a0a0f"` must go, otherwise the dark
background overrides the stylesheet.

## Run

```bash
cd frontend
npm install
npm run dev
```

The backend must be running on port 8000.