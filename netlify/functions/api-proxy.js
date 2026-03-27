exports.handler = async function handler(event) {
  try {
    const rawBase = process.env.API_BASE_URL || "";
    if (!rawBase) {
      return jsonResponse(
        500,
        {
          error:
            "API_BASE_URL is not configured in Netlify environment variables.",
        }
      );
    }

    // Accept API_BASE_URL with or without trailing /api and normalize it.
    const normalizedBase = rawBase.replace(/\/+$/, "").replace(/\/api$/, "");
    const splat = event.pathParameters && event.pathParameters.splat
      ? event.pathParameters.splat
      : "";
    const upstreamUrl = `${normalizedBase}/api/${splat}${event.rawQuery ? `?${event.rawQuery}` : ""}`;

    const requestHeaders = sanitizeRequestHeaders(event.headers || {});
    const hasBody = event.body !== null && event.body !== undefined && event.body !== "";
    const body = hasBody
      ? event.isBase64Encoded
        ? Buffer.from(event.body, "base64")
        : event.body
      : undefined;

    const upstream = await fetch(upstreamUrl, {
      method: event.httpMethod,
      headers: requestHeaders,
      body,
    });

    const responseBody = await upstream.text();
    const responseHeaders = sanitizeResponseHeaders(upstream.headers);

    return {
      statusCode: upstream.status,
      headers: responseHeaders,
      body: responseBody,
    };
  } catch (error) {
    return jsonResponse(502, {
      error: "Failed to reach upstream API",
      details: error && error.message ? error.message : String(error),
    });
  }
};

function sanitizeRequestHeaders(headers) {
  const allowed = {};
  const skip = new Set(["host", "content-length", "x-forwarded-for", "x-forwarded-proto"]);
  for (const [key, value] of Object.entries(headers)) {
    if (!value) continue;
    const lower = key.toLowerCase();
    if (skip.has(lower)) continue;
    allowed[key] = value;
  }
  return allowed;
}

function sanitizeResponseHeaders(headers) {
  const out = {};
  for (const [key, value] of headers.entries()) {
    out[key] = value;
  }
  // Ensure browser can consume proxied API responses.
  out["access-control-allow-origin"] = "*";
  return out;
}

function jsonResponse(statusCode, payload) {
  return {
    statusCode,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "access-control-allow-origin": "*",
    },
    body: JSON.stringify(payload),
  };
}
