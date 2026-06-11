import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        neon: {
          DEFAULT: '#00ff41',
          600:     '#00cc33',
          900:     '#007a1e',
        },
        matrix: {
          bg:      '#080e08',
          card:    '#0d1a0d',
          raised:  '#112211',
          border:  '#1a3a1a',
          input:   '#060e06',
          text:    '#a0d4a0',
          muted:   '#4a7a4a',
          error:   '#ff4141',
          warning: '#ffb800',
          info:    '#00d4ff',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Cascadia Code', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
};

export default config;
