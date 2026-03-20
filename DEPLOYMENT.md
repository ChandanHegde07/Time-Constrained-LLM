# Netlify Deployment Guide

## Prerequisites

1. **Netlify Account**: Sign up at [netlify.com](https://netlify.com)
2. **Netlify CLI**: Install globally with `npm install -g netlify-cli`
3. **Google API Key**: Get your API key from Google AI Studio

## Deployment Steps

### 1. Install Netlify CLI (if not already installed)

```bash
npm install -g netlify-cli
```

### 2. Login to Netlify

```bash
netlify login
```

### 3. Initialize Netlify Project

```bash
netlify init
```

Follow the prompts to:
- Create a new site or link to an existing one
- Set up build settings

### 4. Configure Environment Variables

Set up environment variables in Netlify:

```bash
netlify env:set LLM_API_KEY "your_google_api_key_here"
netlify env:set ENVIRONMENT "production"
```

Or set them via the Netlify Dashboard:
1. Go to your site settings
2. Navigate to "Build & deploy" > "Environment"
3. Add the following variables:
   - `LLM_API_KEY`: Your Google API key
   - `ENVIRONMENT`: `production`
   - `LLM_MODEL`: `gemini-2.5-flash-lite` (or your preferred model)

### 5. Deploy to Netlify

For production deployment:

```bash
netlify deploy --prod
```

For preview deployment:

```bash
netlify deploy
```

## Project Structure

```
.
├── netlify/
│   └── functions/
│       └── api.py        # Netlify Functions entry point
├── app.py                # Flask application
├── config.py             # Configuration management
├── core/                 # Core modules
│   ├── evaluator.py
│   ├── llm.py
│   ├── router.py
│   └── timer.py
├── static/               # Static assets
│   ├── app.js
│   └── styles.css
├── templates/            # HTML templates
│   ├── experiment.html
│   └── home.html
├── netlify.toml          # Netlify configuration
├── requirements.txt      # Python dependencies
└── .env.example          # Environment variable template
```

## Environment Variables

### Required Variables

- `LLM_API_KEY`: Your Google API key for Gemini models

### Optional Variables

- `ENVIRONMENT`: Set to `production` for Netlify deployment
- `LLM_MODEL`: Model to use (default: `gemini-2.5-flash-lite`)
- `LLM_MAX_TOKENS`: Maximum tokens per response (default: 500)
- `LLM_TEMPERATURE`: Temperature for generation (default: 0.7)
- `TIME_LIMITS`: Comma-separated time limits in seconds (default: 1,2,3,5,10,15,30)
- `LOG_LEVEL`: Logging level (default: INFO)

See `.env.example` for all available configuration options.

## Important Notes

### Serverless Limitations

1. **No Background Threads**: Experiments run synchronously in serverless functions
2. **Cold Starts**: First request may be slower due to function initialization
3. **State Management**: Experiment state is in-memory and resets on cold starts
4. **Timeout**: Maximum execution time is 10 seconds for Netlify Functions (free tier)

### Performance Considerations

- Keep experiment task counts very low (recommended: 1-5 tasks)
- Use shorter time limits for faster execution
- Monitor function execution time in Netlify dashboard

### Netlify Functions Limitations

- **Free Tier**: 125,000 function requests per month
- **Execution Time**: 10 seconds maximum per invocation
- **Memory**: 1024 MB maximum
- **Payload Size**: 6 MB maximum

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are in `requirements.txt`
2. **API Key Errors**: Verify `LLM_API_KEY` is set in Netlify environment variables
3. **Timeout Errors**: Reduce task count or time limits significantly
4. **Memory Errors**: Optimize data processing or reduce batch sizes

### Debugging

1. Check Netlify function logs:
   ```bash
   netlify logs
   ```

2. Test locally with Netlify CLI:
   ```bash
   netlify dev
   ```

3. Verify environment variables:
   ```bash
   netlify env:list
   ```

## Custom Domain (Optional)

1. Go to your Netlify site dashboard
2. Navigate to "Domain settings"
3. Add your custom domain
4. Update DNS records as instructed

## Monitoring

- **Netlify Dashboard**: Monitor function executions, errors, and performance
- **Logs**: Access real-time logs via Netlify dashboard or CLI
- **Analytics**: Track usage and performance metrics

## Support

For issues or questions:
1. Check Netlify documentation: https://docs.netlify.com
2. Review Flask documentation: https://flask.palletsprojects.com
3. Check project README.md for additional details

## Migration from Vercel

If migrating from Vercel:
1. Remove `vercel.json` and `api/index.py`
2. Create `netlify.toml` and `netlify/functions/api.py`
3. Update environment variables in Netlify dashboard
4. Deploy using `netlify deploy --prod`
