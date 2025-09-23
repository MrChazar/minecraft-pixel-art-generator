import { BadRequestException, Body, Controller, Get, HttpCode, HttpStatus, Post, Query, Request, UseGuards } from "@nestjs/common";
import { ApiBearerAuth, ApiOperation, ApiResponse, ApiTags } from "@nestjs/swagger";

import { AuthService } from "./auth.service";
import { LoginResponseDto } from "./dto/login-response.dto";
import { LoginDto } from "./dto/login.dto";
import { RegisterDto } from "./dto/register.dto";
import { AuthGuard } from "./auth.guard";
import { JwtService } from "@nestjs/jwt";
import { RegisterResponseDto } from "./dto/register-response.dto";

@Controller("auth")
@ApiTags("auth")
export class AuthController {
  constructor(private readonly authService: AuthService,
    private readonly jwt: JwtService,
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
    return request.user;
  }

  @ApiOperation({
    summary: "Verify email",
  })
  @ApiResponse({
    status: 200,
    description: "Verify email with token send to the user email",
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
}
