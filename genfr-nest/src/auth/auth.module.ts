import { Module } from "@nestjs/common";

import { UserModule } from "../user/user.module";
import { AuthController } from "./auth.controller";
import { AuthService } from "./auth.service";
import { JwtModule } from "@nestjs/jwt";
import { EmailModule } from "../email/email.module";

@Module({
  controllers: [AuthController],
  providers: [AuthService],
  imports: [UserModule, JwtModule.register({
    global: true,
    secret: process.env.JWTSECRET,
    signOptions: { expiresIn: '3600s' },
  }),
  EmailModule

],
  exports: [AuthService],
})
export class AuthModule {}
