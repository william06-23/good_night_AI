import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import StoryApp from './StoryApp.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <StoryApp />
  </StrictMode>,
)
