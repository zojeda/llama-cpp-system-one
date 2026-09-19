import { UnprocessableEntityError, noul } from "@typesafe-ai/sdk";
import { client } from "./client.mjs";

try {
  // The SDK forwards extensions, but this server accepts only a single read.
  await client.systemOne({
    state: "Ground granulated blast furnace slag is used in concrete.",
    questions: { is_scm: noul("Is the material an SCM?") },
    steps: 2,
  });
  throw new Error("Expected this server to reject steps=2 with HTTP 422");
} catch (error) {
  if (!(error instanceof UnprocessableEntityError)) {
    throw error;
  }
  console.log(JSON.stringify({
    name: error.name,
    status: error.status,
    requestId: error.requestId,
    details: error.body,
  }, null, 2));
}
