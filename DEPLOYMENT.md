# Vercel Deployment Guide

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI**: Install globally with `npm i -g vercel`
3. **Google API Key**: Get your API key from Google AI Studio

## Deployment Steps

### 1. Install Vercel CLI (if not already installed)

```bash
npm i -g vercel
```

### 2. Login to Vercel

```bash
vercel login
```

### 3. Configure Environment Variables

Before deploying, you need to set up environment variables in Vercel:

```bash
vercel env add LLM_API_KEY
```

When prompted, enter your Google API key.

Or set them via the Vercel Dashboard:
1. Go to your project settings
2. Navigate to "Environment Variables"
3. Add the following variables:
   - `LLM_API_KEY`: Your Google API key
   - `ENVIRONMENT`: `production`
   - `LLM_MODEL`: `gemini-2.5-flash-lite` (or your preferred model)

### 4. Deploy to Vercel

```bash
vercel --prod
```

Or for preview deployment:

```bash
vercel
```

## Project Structure

```
.
├── api/
│   └── index.py          # Vercel serverless entry point
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
├── vercel.json           # Vercel configuration
├── requirements.txt      # Python dependencies
└── .env.example          # Environment variable template
```

## Environment Variables

### Required Variables

- `LLM_API_KEY`: Your Google API key for Gemini models

### Optional Variables

- `ENVIRONMENT`: Set to `production` for Vercel deployment
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
4. **Timeout**: Maximum execution time is 300 seconds (5 minutes)

### Performance Considerations

- Keep experiment task counts reasonable (recommended: 10-20 tasks)
- Use shorter time limits for faster execution
- Monitor function execution time in Vercel dashboard

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are in `requirements.txt`
2. **API Key Errors**: Verify `LLM_API_KEY` is set in Vercel environment variables
3. **Timeout Errors**: Reduce task count or time limits
4. **Memory Errors**: Optimize data processing or reduce batch sizes

### Debugging

1. Check Vercel function logs:
   ```bash
   vercel logs
   ```

2. Test locally with Vercel CLI:
   ```bash
   vercel dev
   ```

3. Verify environment variables:
   ```bash
   vercel env ls
   ```

## Custom Domain (Optional)

1. Go to your Vercel project dashboard
2. Navigate to "Settings" > "Domains"
3. Add your custom domain
4. Update DNS records as instructed

## Monitoring

- **Vercel Dashboard**: Monitor function executions, errors, and performance
- **Logs**: Access real-time logs via Vercel dashboard or CLI
- **Analytics**: Track usage and performance metrics

## Support

For issues or questions:
1. Check Vercel documentation: https://vercel.com/docs
2. Review Flask documentation: https://flask.palletsprojects.com
3. Check project README.md for additional details
