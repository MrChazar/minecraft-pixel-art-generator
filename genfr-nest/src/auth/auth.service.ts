import * as bcrypt from "bcrypt";

import { ConflictException, Injectable, UnauthorizedException } from "@nestjs/common";

import { UserService } from "../user/user.service";
import { LoginResponseDto } from "./dto/login-response.dto";
import { RegisterResponseDto } from "./dto/register-response.dto";
import { RegisterDto } from "./dto/register.dto";
import { JwtService } from "@nestjs/jwt";
import { EmailService } from "../email/email.service";

@Injectable()
export class AuthService {
  constructor(private readonly userService: UserService,
    private jwtService: JwtService,
    private mailservice: EmailService
  ) {}

  private static readonly EXPIRY_TIME_MS = Number.parseInt(
    process.env.EXPIRY_TIME_MS ?? "3600000",
  );


  async signIn(email: string, password: string): Promise<LoginResponseDto> {
    const user = await this.userService.findOne(email);
    if (
      user === null ||
      !user.is_enabled ||
      !(await bcrypt.compare(password, user.password).catch(() => false))
    ) {
      throw new UnauthorizedException();
    }
    const payload = {email: user.email , role:user.role};
    await this.mailservice.sendTest(user.email)
    return { access_token: await this.jwtService.signAsync(payload) };
  }

  async signUp(userData: RegisterDto): Promise<RegisterResponseDto> {
    const checkExists = await this.userService.findOne(userData.email);
    if (
      !(checkExists === null))
     {
      throw new ConflictException("User with this email already exists");
    }


    const user = await this.userService.createUser(
      {email:userData.email,
      password:userData.password,
      about_me:userData.about_me ?? '',
      name:userData.name ?? ''}
    );

    const payload = {sub:user.email , username: user.email };
    return { access_token: await this.jwtService.signAsync(payload) };
  }
}
