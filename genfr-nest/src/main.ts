/* eslint-disable unicorn/prefer-module */
import { NestFactory } from "@nestjs/core";
import type { NestExpressApplication } from '@nestjs/platform-express';

import { AppModule } from "./app.module";
import { join } from "node:path";
import { DocumentBuilder, SwaggerModule } from "@nestjs/swagger";
import { ValidationPipe } from "@nestjs/common";

async function bootstrap() {
  const app = await NestFactory.create<NestExpressApplication>(AppModule);

  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      transform: true,
      forbidUnknownValues: true,
      forbidNonWhitelisted: true,
    }),
  );

  app.useStaticAssets(join(__dirname, '..', 'media'), { prefix: '/media/' });
  const config = new DocumentBuilder()
    .setTitle('Generator Frajdy Api')
    .setDescription('API documentation Swagger')
    .setVersion('1.0')
    .addTag("image", "Zarządzanie generowanymi obrazami")
    .addBearerAuth({ type: "http", scheme: "bearer", bearerFormat: "JWT" },
      "access-token",)
    .build();
  
  const document = SwaggerModule.createDocument(app, config);
  SwaggerModule.setup('api', app, document);

  await app.listen(process.env.PORT ?? 3000);
}
bootstrap();
