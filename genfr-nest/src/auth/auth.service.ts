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
      !(await bcrypt.compare(password, user.password).catch(() => false)) ||
      !user.isVerified
    ) {
      throw new UnauthorizedException();
    }
    const payload = {email: user.email , role:user.role};
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
    const token = this.jwtService.sign(
      { sub: user.email, purpose: 'email-verify' },
      { expiresIn: '1h' }
    );

    const verifyUrl = `${process.env.FRONTEND_URL}/auth/verify-email?token=${token}`

    await this.mailservice.sendVerificationMail(user.email, verifyUrl)

    const payload = {sub:user.email , username: user.email };
    return { message: 'The verification email has been sent to the provided email. Please check you mailbox' };
  }

  async markVerified(userEmail: string){
    await this.userService.verifyUser(userEmail)
  }

  
}
