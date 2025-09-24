import { BadRequestException, Body, Controller, Get, HttpCode, HttpStatus, Post, Query, Request, UnauthorizedException, UseGuards } from "@nestjs/common";
import { ApiBearerAuth, ApiOperation, ApiResponse, ApiTags } from "@nestjs/swagger";

import { AuthService } from "./auth.service";
import { LoginResponseDto } from "./dto/login-response.dto";
import { LoginDto } from "./dto/login.dto";
import { RegisterDto } from "./dto/register.dto";
import { AuthGuard } from "./auth.guard";
import { JwtService } from "@nestjs/jwt";
import { RegisterResponseDto } from "./dto/register-response.dto";
import { EmailService } from "../email/email.service";
import { UserService } from "../user/user.service";
import { ForgotPasswordDto } from "./dto/forgot-password.dto";
import { ResetPasswordDto } from "./dto/reset-password.dto";

@Controller("auth")
@ApiTags("auth")
export class AuthController {
  
  constructor(private readonly authService: AuthService,
    private readonly userService: UserService,
    private readonly jwt: JwtService,
    private mailservice: EmailService
  ) {}

  @ApiOperation({
    summary: "Log in with an existing account",
  })
  @ApiResponse({
    status: 200,
    description: "Logged in",
  })
  @ApiResponse({
    status: 401,
    description: "Invalid credentials or account disabled",
  })
  @HttpCode(HttpStatus.OK)
  @Post("login")
  async signIn(@Body() signInDto: LoginDto): Promise<LoginResponseDto> {
    return await this.authService.signIn(signInDto.email, signInDto.password);
  }

  @ApiOperation({
    summary: "Register a new user account",
  })
  @ApiResponse({
    status: 201,
    description: "Account created",
  })
  @ApiResponse({
    status: 409,
    description: "User with the given email already exists",
  })
  @ApiResponse({
    status: 400,
    description: "Invalid request data",
  })
  @ApiResponse({
    status: 500,
    description: "Internal server error",
  })
  @HttpCode(HttpStatus.CREATED)
  @Post("register")
  async signUp(@Body() signUpDto: RegisterDto): Promise<RegisterResponseDto> {
    return await this.authService.signUp({
      email: signUpDto.email,
      password: signUpDto.password,
    });
  }

  @ApiOperation({
    summary: "Get logged in user",
  })
  @ApiResponse({
    status: 200,
    description: "Get a profile of currently logged in user",
  })
  @UseGuards(AuthGuard)
  @Get('profile')
  @ApiBearerAuth("access-token")
  getProfile(@Request() request) {
    if (request.user === null || request.user === undefined){
      throw new UnauthorizedException("You need to be logged in for that")
    }
    return request.user;
  }

  @ApiOperation({
    summary: "Verify email",
  })
  @ApiResponse({
    status: 200,
    description: "Verify email with token sent to the user email",
  })
  @Get('verify-email')
  async verify(@Query('token') token: string) {

    try {
      const payload = this.jwt.verify(token);
      if (payload.purpose !== 'email-verify') throw new BadRequestException("Invalid token");
      await this.authService.markVerified(payload.sub);
      return { message: 'Email verified successfully' };
    } catch {
      throw new BadRequestException('Invalid or expired token');
    }
  }

  @ApiOperation({
    summary: "Reset password",
  })
  @ApiResponse({
    status: 200,
    description: "Reset password with token sent to the user email",
  })
  @Post('reset-password')
  async reset(@Body() dto: ResetPasswordDto) {
    try {
      const payload = this.jwt.verify(dto.token);
      if (payload.purpose !== 'reset-password') throw new Error();

      await this.authService.changePassword(payload.sub, dto.newPassword);

      return { message: 'Password updated successfully' };
    } catch {
      throw new BadRequestException('Invalid or expired token');
    }
  }

  @ApiOperation({
    summary: "forgot password",
  })
  @ApiResponse({
    status: 200,
    description: "Send a link to the provided email in order to reset the account password",
  })
  @Post('forgot-password')
  async forgot(@Body() body: ForgotPasswordDto) {
    const { email } = body;
    const user = await this.userService.findOne(email);
    if (!user) return { message: 'Account with this email does not exist' };

    const token = this.jwt.sign(
      { sub: user.email, purpose: 'reset-password' },
      { expiresIn: '15m' }
    );

    const url = `${process.env.FRONTEND_URL}/reset-password/${token}`;

    await this.mailservice.sendPasswordResetMail(user.email,url)

    return { message: 'If that email exists, a link has been sent' };
  }


}
