import { Module } from "@nestjs/common";
import { AppController } from "./app.controller";
import { AppService } from "./app.service";
import { PrismaModule } from "./prisma/prisma.module";
import { ServeStaticModule } from "@nestjs/serve-static";
import { join } from "node:path";
import { ImageModule } from './image/image.module';
import { AuthModule } from "./auth/auth.module";
import { UserModule } from "./user/user.module";


@Module({
  imports: [PrismaModule,
    ServeStaticModule.forRoot({
      rootPath:join(__dirname,"../..","media"),
      serveRoot:"/media",
    }),
    ImageModule,
    AuthModule,
    UserModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
